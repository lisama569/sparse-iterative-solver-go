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
	n := len(x)
	res := &Result{Status: StatusMaxIterations, X: x}

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

	r := make([]float64, n)
	p := make([]float64, n)
	Ap := make([]float64, n)
	z := make([]float64, n)

	Ax, err := A.MatVec(x)
	if err != nil {
		return invalidFrom(err, x)
	}
	bNorm := vnorm(b)
	for i := 0; i < n; i++ {
		r[i] = b[i] - Ax[i]
	}
	rel := safeRelNorm(r, bNorm)
	res.History = append(res.History, rel)
	if rel <= cfg.Tolerance {
		res.Status = StatusConverged
		res.FinalRelRes = rel
		return res
	}

	applyP(z, r)
	copy(p, z)
	rz := vdot(r, z)
	if rz <= 0 || math.IsNaN(rz) {
		res.Status = StatusBreakdown
		res.ErrorMessage = "preconditioned residual inner product non-positive: " + floatStr(rz)
		res.FinalRelRes = rel
		return res
	}

	for iter := 1; iter <= cfg.MaxIterations; iter++ {
		select {
		case <-ctx.Done():
			res.Status = StatusCanceled
			res.ErrorMessage = ctx.Err().Error()
			res.FinalRelRes = rel
			res.Iterations = iter - 1
			return res
		default:
		}
		if !finiteVec(r) || !finiteVec(p) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite residual or search direction"
			res.FinalRelRes = math.NaN()
			res.Iterations = iter - 1
		}

		Ap, err = A.MatVec(p)
		if err != nil {
			return invalidFrom(err, x)
		}
		pAp := vdot(p, Ap)
		if math.IsNaN(pAp) || math.IsInf(pAp, 0) || pAp <= 0 {
			// Non-positive curvature: A is not SPD along p, or rounding
			// destroyed conjugacy. CG cannot continue reliably.
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-positive curvature p^T A p = " + floatStr(pAp)
			res.FinalRelRes = rel
			res.Iterations = iter - 1
			return res
		}
		alpha := rz / pAp
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
			r[i] -= alpha * Ap[i]
		}
		if !finiteVec(r) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite residual after update"
			res.FinalRelRes = math.NaN()
			res.Iterations = iter
			return res
		}
		rel = safeRelNorm(r, bNorm)
		res.History = append(res.History, rel)
		if rel <= cfg.Tolerance {
			res.Status = StatusConverged
			res.FinalRelRes = rel
			res.Iterations = iter
			return res
		}
		applyP(z, r)
		rzNew := vdot(r, z)
		if rzNew <= 0 {
			res.Status = StatusBreakdown
			res.ErrorMessage = "preconditioned residual inner product non-positive after update"
			res.FinalRelRes = rel
			res.Iterations = iter
			return res
		}
		beta := rzNew / rz
		if math.IsNaN(beta) || math.IsInf(beta, 0) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "non-finite beta in CG update"
			res.FinalRelRes = rel
			res.Iterations = iter
			return res
		}
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rz = rzNew
	}
	res.FinalRelRes = rel
	return res
}
