package solver

import (
	"context"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// solveCG runs (optionally Jacobi-preconditioned) conjugate gradients.
//
// CG assumes a symmetric positive definite operator. On a matrix that is
// symmetric but not positive definite (or with a poor preconditioner) the
// method breaks down when p^T A p <= 0; that is reported as
// StatusBreakdown rather than convergence.
func solveCG(ctx context.Context, A *sparse.CSR, b, x []float64, cfg Config) *Result {
	c, term := newCGCheckpoint(A, b, x, cfg)
	if term != nil {
		return term
	}
	res, _ := cgRun(ctx, A, b, cfg, c, cfg.MaxIterations)
	return res
}

// newCGCheckpoint builds the initial checkpoint for a CG solve from the
// starting vector x. It returns a non-nil Result when the solve already
// terminates during initialization (invalid input detected by MatVec).
func newCGCheckpoint(A *sparse.CSR, b, x []float64, cfg Config) (*Checkpoint, *Result) {
	n := len(x)
	Ax, err := A.MatVec(x)
	if err != nil {
		return nil, invalidFrom(err, x)
	}
	r := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = b[i] - Ax[i]
	}
	rel := safeRelNorm(r, vnorm(b))
	c := &Checkpoint{
		Method:       MethodCG,
		N:            n,
		Tolerance:    cfg.Tolerance,
		Precondition: cfg.Precondition,
		Rel:          rel,
		History:      []float64{rel},
		X:            x,
		R:            r,
	}
	return c, nil
}

// cgRun advances CG from checkpoint c by at most budget iterations (and never
// past cfg.MaxIterations total). It returns the result and, when the status
// is StatusPaused or StatusCanceled, a checkpoint that resumes the solve with
// bit-identical trajectory to an uninterrupted run.
func cgRun(ctx context.Context, A *sparse.CSR, b []float64, cfg Config, c *Checkpoint, budget int) (*Result, *Checkpoint) {
	n := len(c.X)
	res := &Result{Status: StatusMaxIterations, X: c.X, Tolerance: cfg.Tolerance}

	var diag []float64
	if cfg.Precondition {
		diag = A.Diagonal()
	}
	applyP := func(z, r []float64) {
		if cfg.Precondition {
			for i := 0; i < n; i++ {
				z[i] = r[i] / diag[i]
			}
			return
		}
		copy(z, r)
	}

	bNorm := vnorm(b)
	rel := c.Rel
	res.History = append(res.History, c.History...)
	iters := c.Iterations

	r := c.R
	p := c.P
	z := make([]float64, n)

	// snapshot copies the live state so the caller may keep using the
	// returned Result while resuming later from the checkpoint.
	snapshot := func() *Checkpoint {
		return &Checkpoint{
			Method:       MethodCG,
			N:            n,
			Iterations:   iters,
			Tolerance:    cfg.Tolerance,
			Precondition: cfg.Precondition,
			Rel:          rel,
			History:      append([]float64(nil), res.History...),
			X:            append([]float64(nil), c.X...),
			R:            append([]float64(nil), r...),
			P:            append([]float64(nil), p...),
			Rz:           c.Rz,
		}
	}

	// Initialization of the search direction happens once, either for a
	// fresh solve or never (a resumed checkpoint already carries p and rz).
	rz := c.Rz
	if p == nil {
		if rel <= cfg.Tolerance {
			res.Status = StatusConverged
			res.FinalRelRes = rel
			res.Iterations = 0
			return res, nil
		}
		applyP(z, r)
		p = append([]float64(nil), z...)
		c.P = p
		rz = vdot(r, z)
		c.Rz = rz
		if rz <= 0 || math.IsNaN(rz) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "preconditioned residual inner product non-positive: " + floatStr(rz)
			res.FinalRelRes = rel
			return res, nil
		}
	}

	startIters := iters
	for {
		if iters >= cfg.MaxIterations {
			// Total step budget exhausted: terminal, not resumable.
			res.Status = StatusMaxIterations
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, nil
		}
		if iters-startIters >= budget {
			res.Status = StatusPaused
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, snapshot()
		}
		select {
		case <-ctx.Done():
			res.Status = StatusCanceled
			res.ErrorMessage = ctx.Err().Error()
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, snapshot()
		default:
		}
		if !finiteVec(r) || !finiteVec(p) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite residual or search direction"
			res.FinalRelRes = math.NaN()
			res.Iterations = iters
			return res, nil
		}

		Ap, err := A.MatVec(p)
		if err != nil {
			return invalidFrom(err, c.X), nil
		}
		pAp := vdot(p, Ap)
		if math.IsNaN(pAp) || math.IsInf(pAp, 0) || pAp <= 0 {
			// Non-positive curvature: A is not SPD along p, or rounding
			// destroyed conjugacy. CG cannot continue reliably.
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-positive curvature p^T A p = " + floatStr(pAp)
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, nil
		}
		alpha := rz / pAp
		for i := 0; i < n; i++ {
			c.X[i] += alpha * p[i]
			r[i] -= alpha * Ap[i]
		}
		iters++
		if !finiteVec(r) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite residual after update"
			res.FinalRelRes = math.NaN()
			res.Iterations = iters
			return res, nil
		}
		rel = safeRelNorm(r, bNorm)
		res.History = append(res.History, rel)
		if rel <= cfg.Tolerance {
			res.Status = StatusConverged
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, nil
		}
		applyP(z, r)
		rzNew := vdot(r, z)
		if rzNew <= 0 {
			res.Status = StatusBreakdown
			res.ErrorMessage = "preconditioned residual inner product non-positive after update"
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, nil
		}
		beta := rzNew / rz
		if math.IsNaN(beta) || math.IsInf(beta, 0) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite beta in CG update"
			res.FinalRelRes = rel
			res.Iterations = iters
			return res, nil
		}
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rz = rzNew
		c.Rz = rz
	}
}
