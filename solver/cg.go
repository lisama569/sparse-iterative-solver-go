package solver

import (
	"context"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// cgState is a resumable conjugate-gradients checkpoint.
//
// Every checkpoint boundary sits between iterations, so (x, r, p, z, rz)
// are always mutually consistent: r = b - A*x as last computed, p is the
// current search direction and rz = r·z. A paused or canceled segment can
// therefore resume with bit-identical arithmetic; concatenating segments is
// the same computation as a one-shot solve.
type cgState struct {
	cfg  Config
	n    int
	diag []float64 // nil without Jacobi preconditioning
	bn   float64   // ||b||

	x, r, p, Ap, z []float64
	rz             float64

	iter      int
	rel       float64 // last MEASURED relative residual
	history   []float64
	errDetail string
}

// newCGState performs CG initialization (initial residual, first
// preconditioned direction). The returned status is terminal (convergence at
// iteration 0, immediate breakdown) or StatusPaused acting as the
// "ready to step" sentinel.
func newCGState(cfg Config, A *sparse.CSR, b, x0 []float64) (*cgState, Status, error) {
	n := len(x0)
	s := &cgState{cfg: cfg, n: n}
	if cfg.Precondition {
		s.diag = A.Diagonal()
	}
	s.x = make([]float64, n)
	copy(s.x, x0)

	Ax, err := A.MatVec(s.x)
	if err != nil {
		return s, StatusInvalidInput, err
	}
	s.bn = vnorm(b)
	s.r = make([]float64, n)
	for i := 0; i < n; i++ {
		s.r[i] = b[i] - Ax[i]
	}
	s.rel = safeRelNorm(s.r, s.bn)
	s.history = append(s.history, s.rel)
	if s.rel <= cfg.Tolerance {
		s.errDetail = "initial guess already satisfies the residual requirement"
		return s, StatusConverged, nil
	}

	s.p = make([]float64, n)
	s.z = make([]float64, n)
	s.Ap = make([]float64, n)
	s.applyPrecond(s.z, s.r)
	copy(s.p, s.z)
	s.rz = vdot(s.r, s.z)
	if math.IsNaN(s.rz) || s.rz <= 0 {
		s.errDetail = "preconditioned residual inner product non-positive: " + floatStr(s.rz)
		return s, StatusBreakdown, nil
	}
	return s, StatusPaused, nil // sentinel: ready, not actually paused
}

func (s *cgState) applyPrecond(z, r []float64) {
	if s.cfg.Precondition {
		for i := 0; i < s.n; i++ {
			z[i] = r[i] / s.diag[i]
		}
		return
	}
	copy(z, r)
}

func (s *cgState) iterations() int { return s.iter }
func (s *cgState) relRes() float64 { return s.rel }
func (s *cgState) method() Method  { return MethodCG }

// advance performs up to maxSteps CG iterations. When the step allowance is
// exhausted it returns StatusMaxIterations if atTotalCap (the global budget
// is used up) or StatusPaused (resumable) otherwise.
func (s *cgState) advance(ctx context.Context, A *sparse.CSR, b []float64, maxSteps int, atTotalCap bool) Status {
	for k := 0; k < maxSteps; k++ {
		if err := ctx.Err(); err != nil {
			s.errDetail = err.Error()
			return StatusCanceled
		}
		if !finiteVec(s.r) || !finiteVec(s.p) {
			s.errDetail = "non-finite residual or search direction"
			s.rel = math.NaN()
			return StatusBreakdown
		}
		var err error
		s.Ap, err = A.MatVec(s.p)
		if err != nil {
			s.errDetail = err.Error()
			return StatusInvalidInput
		}
		pAp := vdot(s.p, s.Ap)
		if math.IsNaN(pAp) || math.IsInf(pAp, 0) || pAp <= 0 {
			// Non-positive curvature: A is not SPD along p, or rounding
			// destroyed conjugacy. CG cannot continue reliably.
			s.errDetail = "non-positive curvature p^T A p = " + floatStr(pAp)
			return StatusBreakdown
		}
		alpha := s.rz / pAp
		for i := 0; i < s.n; i++ {
			s.x[i] += alpha * s.p[i]
			s.r[i] -= alpha * s.Ap[i]
		}
		if !finiteVec(s.r) {
			s.errDetail = "non-finite residual after update"
			s.rel = math.NaN()
			s.iter++
			return StatusBreakdown
		}
		s.rel = safeRelNorm(s.r, s.bn)
		s.history = append(s.history, s.rel)
		s.iter++
		if s.rel <= s.cfg.Tolerance {
			return StatusConverged
		}
		s.applyPrecond(s.z, s.r)
		rzNew := vdot(s.r, s.z)
		if rzNew <= 0 {
			s.errDetail = "preconditioned residual inner product non-positive after update"
			return StatusBreakdown
		}
		beta := rzNew / s.rz
		if math.IsNaN(beta) || math.IsInf(beta, 0) {
			s.errDetail = "non-finite beta in CG update"
			return StatusBreakdown
		}
		for i := 0; i < s.n; i++ {
			s.p[i] = s.z[i] + beta*s.p[i]
		}
		s.rz = rzNew
	}
	// Step allowance exhausted. The residual is measured every iteration,
	// so s.rel is an up-to-date measured value.
	if atTotalCap {
		return StatusMaxIterations
	}
	return StatusPaused
}

// measure recomputes ||b - A*x||/||b|| from scratch (CG does so every step,
// so this is just the cached value; kept for driver uniformity).
func (s *cgState) measure(_ *sparse.CSR, _ []float64) float64 { return s.rel }

// checkCompatible verifies that a continuation uses the same problem shape
// and the same decisive configuration as the segment that produced p.
func (s *cgState) checkCompatible(cfg Config, n int) error {
	if s.n != n {
		return ErrProgressMismatch
	}
	if cfg.Method != MethodCG || !s.cfg.match(cfg) {
		return ErrProgressMismatch
	}
	return nil
}

// clone returns a deep copy so the Progress handed to a caller cannot be
// mutated when the caller chooses to continue from the same checkpoint
// more than once or archives the result alongside further segments.
func (s *cgState) clone() *cgState {
	c := *s
	c.x = cloneFloats(s.x)
	c.r = cloneFloats(s.r)
	c.p = cloneFloats(s.p)
	c.Ap = cloneFloats(s.Ap)
	c.z = cloneFloats(s.z)
	c.diag = cloneFloats(s.diag)
	c.history = append([]float64(nil), s.history...)
	return &c
}
