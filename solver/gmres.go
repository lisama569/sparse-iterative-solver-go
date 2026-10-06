package solver

import (
	"context"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// gmresState is a resumable restarted GMRES(m) checkpoint using modified
// Gram-Schmidt Arnoldi and Givens-rotation least squares.
//
// A checkpoint can sit either at a cycle boundary (j == 0, r is the exact
// residual of x) or INSIDE a cycle (0 < j < m: the Arnoldi basis V, upper
// Hessenberg H and Givens state are retained). The stored x always equals
// the committed cycle-start iterate; the best-fit correction of an
// unfinished cycle is computed on demand for residual measurement and again
// at the terminal step, so resuming an unfinished cycle performs exactly
// the same arithmetic as a one-shot solve.
type gmresState struct {
	cfg  Config
	n, m int
	diag []float64 // nil without Jacobi (right) preconditioning
	bn   float64   // ||b||

	x []float64 // committed solution at the current cycle start
	r []float64 // exact residual at the current cycle start

	V         [][]float64 // m+1 Krylov vectors
	H         []float64   // (m+1) x m, column-major, stride m
	cs, sn, g []float64
	j         int // completed Arnoldi steps in the current cycle
	iter      int // total matrix-vector products
	rel       float64
	history   []float64
	errDetail string
}

// newGMRESState validates the restart length and measures the initial
// residual. StatusConverged means the initial guess already satisfies the
// tolerance; otherwise StatusPaused is the "ready to step" sentinel.
func newGMRESState(cfg Config, A *sparse.CSR, b, x0 []float64) (*gmresState, Status, error) {
	n := len(x0)
	m := cfg.Restart
	if m > n {
		m = n
	}
	s := &gmresState{cfg: cfg, n: n, m: m}
	if m <= 0 {
		s.errDetail = "GMRES restart length must be positive"
		return s, StatusInvalidInput, nil
	}
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
	return s, StatusPaused, nil
}

func (s *gmresState) iterations() int { return s.iter }
func (s *gmresState) relRes() float64 { return s.rel }
func (s *gmresState) method() Method  { return MethodGMRES }
func (s *gmresState) detail() string  { return s.errDetail }

func (s *gmresState) applyMinv(v []float64) []float64 {
	out := make([]float64, s.n)
	if s.cfg.Precondition {
		for i := 0; i < s.n; i++ {
			out[i] = v[i] / s.diag[i]
		}
	} else {
		copy(out, v)
	}
	return out
}

// amp computes w = A M^-1 v.
func (s *gmresState) amp(A *sparse.CSR, v []float64) ([]float64, error) {
	return A.MatVec(s.applyMinv(v))
}

// startCycle initializes a fresh Arnoldi cycle from the exact residual r.
func (s *gmresState) startCycle() {
	beta := vnorm(s.r)
	s.V = make([][]float64, s.m+1)
	s.V[0] = scale(1/beta, s.r)
	s.H = make([]float64, (s.m+1)*s.m)
	s.cs = make([]float64, s.m)
	s.sn = make([]float64, s.m)
	s.g = make([]float64, s.m+1)
	s.g[0] = beta
	s.j = 0
}

// trialX returns x + M^-1 * (V_k y), the best-fit iterate of the current
// cycle, WITHOUT committing it.
func (s *gmresState) trialX(k int) []float64 {
	y := backSolve(s.H, s.g, k, s.m)
	x := cloneFloats(s.x)
	return axpyIntoX(x, s.applyMinv(mulV(s.V, y, k)))
}

// measure computes the measured relative residual of a candidate iterate.
func (s *gmresState) measure(A *sparse.CSR, b, x []float64) float64 {
	Ax, _ := A.MatVec(x)
	rr := make([]float64, s.n)
	for i := 0; i < s.n; i++ {
		rr[i] = b[i] - Ax[i]
	}
	return safeRelNorm(rr, s.bn)
}

// boundaryRitual performs the pause/terminal/cancel bookkeeping at a cycle
// boundary or in the middle of a cycle: form the best-fit iterate, MEASURE
// the true residual and record one explicit history point. The committed
// cycle-start state is preserved (pass commit=false) so an unfinished cycle
// can resume bit-identically; a true restart (commit=true) resets the cycle
// from the measured residual.
func (s *gmresState) boundaryRitual(A *sparse.CSR, b []float64, commit bool) float64 {
	xt := s.trialX(s.j)
	rel := s.measure(A, b, xt)
	s.rel = rel
	s.history = append(s.history, rel)
	if commit {
		s.x = xt
		Ax, _ := A.MatVec(s.x)
		for i := 0; i < s.n; i++ {
			s.r[i] = b[i] - Ax[i]
		}
		s.V = nil
		s.j = 0
	}
	return rel
}

// advance performs up to maxSteps matrix-vector steps. atTotalCap reports
// whether this allowance reaches cfg.MaxIterations (terminal budget) rather
// than a per-segment cap (pause).
func (s *gmresState) advance(ctx context.Context, A *sparse.CSR, b []float64, maxSteps int, atTotalCap bool) Status {
	totalCap := s.cfg.MaxIterations

	// Resume inside an unfinished cycle, or start the first one.
	if s.j > 0 {
		// nothing to initialize
	} else {
		beta := vnorm(s.r)
		if beta == 0 || math.IsNaN(beta) {
			// Exact-zero residual at a non-initial point must have been
			// caught when measured; NaN means arithmetic is unusable.
			s.rel = safeRelNorm(s.r, s.bn)
			if s.rel <= s.cfg.Tolerance {
				return StatusConverged
			}
			s.errDetail = "GMRES residual norm not usable: " + floatStr(beta)
			return StatusBreakdown
		}
		s.startCycle()
	}

	stepsLeft := maxSteps
	for stepsLeft > 0 && s.iter < totalCap {
		// Finish the current cycle (possibly resumed mid-cycle), then
		// restart until the allowance is used.
		for s.j < s.m && stepsLeft > 0 && s.iter < totalCap {
			if err := ctx.Err(); err != nil {
				// Keep everything earned: measured best-fit iterate plus a
				// resumable checkpoint of the unfinished cycle.
				s.boundaryRitual(A, b, false)
				s.errDetail = err.Error()
				return StatusCanceled
			}
			w, err := s.amp(A, s.V[s.j])
			if err != nil {
				s.errDetail = err.Error()
				return StatusInvalidInput
			}
			if !finiteVec(w) {
				s.errDetail = "non-finite Arnoldi vector"
				s.rel = math.NaN()
				return StatusBreakdown
			}
			// Modified Gram-Schmidt.
			for i := 0; i <= s.j; i++ {
				hij := vdot(w, s.V[i])
				s.H[i*s.m+s.j] = hij
				w = axpy(-hij, s.V[i], w)
			}
			hnorm := vnorm(w)
			s.H[(s.j+1)*s.m+s.j] = hnorm

			// Apply previous Givens rotations to this column:
			// (a,b) -> (c*a + s*b, -s*a + c*b).
			for i := 0; i < s.j; i++ {
				h1 := s.H[i*s.m+s.j]
				h2 := s.H[(i+1)*s.m+s.j]
				s.H[i*s.m+s.j] = s.cs[i]*h1 + s.sn[i]*h2
				s.H[(i+1)*s.m+s.j] = -s.sn[i]*h1 + s.cs[i]*h2
			}
			h1 := s.H[s.j*s.m+s.j]
			h2 := s.H[(s.j+1)*s.m+s.j]
			denom := math.Hypot(h1, h2)
			if denom == 0 || math.IsNaN(denom) {
				// Lucky breakdown: A-invariant Krylov subspace. Decide on
				// the MEASURED residual of the best-fit iterate.
				resid := math.Abs(s.g[s.j])
				relProj := safeScale(resid, s.bn)
				s.history = append(s.history, relProj)
				s.iter++
				stepsLeft--
				rel := s.measure(A, b, s.trialX(s.j))
				s.rel = rel
				if rel <= s.cfg.Tolerance {
					s.x = s.trialX(s.j)
					return StatusConverged
				}
				s.errDetail = "GMRES breakdown: zero Krylov basis norm without convergence"
				return StatusBreakdown
			}
			// New rotation G_j zeroes H[j+1,j]:
			// (h1,h2) -> (d,0) with d = hypot(h1,h2).
			s.cs[s.j] = h1 / denom
			s.sn[s.j] = h2 / denom
			s.H[s.j*s.m+s.j] = denom
			s.H[(s.j+1)*s.m+s.j] = 0
			gj := s.g[s.j]
			s.g[s.j] = s.cs[s.j]*gj + s.sn[s.j]*s.g[s.j+1]
			s.g[s.j+1] = -s.sn[s.j]*gj + s.cs[s.j]*s.g[s.j+1]

			s.iter++
			stepsLeft--
			relProj := safeScale(math.Abs(s.g[s.j+1]), s.bn)
			s.history = append(s.history, relProj)
			if relProj <= s.cfg.Tolerance {
				// Projected claim: verify with an explicit measurement.
				xt := s.trialX(s.j + 1)
				rel := s.measure(A, b, xt)
				if rel <= s.cfg.Tolerance {
					s.x = xt
					s.rel = rel
					return StatusConverged
				}
				// Projected residual was optimistic: commit the best-fit
				// iterate and start a fresh cycle with the exact residual.
				s.rel = rel
				s.history = append(s.history, rel)
				s.x = xt
				Ax, _ := A.MatVec(s.x)
				for i := 0; i < s.n; i++ {
					s.r[i] = b[i] - Ax[i]
				}
				if s.iter >= totalCap {
					if rel <= s.cfg.Tolerance {
						return StatusConverged
					}
					return StatusMaxIterations
				}
				beta := vnorm(s.r)
				if beta == 0 || math.IsNaN(beta) {
					if rel <= s.cfg.Tolerance {
						return StatusConverged
					}
					s.errDetail = "GMRES residual norm not usable: " + floatStr(beta)
					return StatusBreakdown
				}
				s.startCycle()
				continue
			}
			if hnorm == 0 {
				// Invariant subspace reached without convergence: further
				// restarts cannot change the result.
				s.boundaryRitual(A, b, false)
				s.errDetail = "GMRES breakdown: invariant Krylov subspace before convergence"
				return StatusBreakdown
			}
			s.V[s.j+1] = scale(1/hnorm, w)
			s.j++
		}

		if s.iter >= totalCap || stepsLeft == 0 {
			break
		}

		// Full cycle complete: commit the least-squares update and restart
		// from the explicitly recomputed true residual.
		rel := s.boundaryRitual(A, b, true)
		if rel <= s.cfg.Tolerance {
			return StatusConverged
		}
		beta := vnorm(s.r)
		if beta == 0 || math.IsNaN(beta) {
			s.errDetail = "GMRES residual norm not usable: " + floatStr(beta)
			return StatusBreakdown
		}
		s.startCycle()
	}

	// Allowance exhausted. Measure the best-fit iterate WITHOUT destroying
	// an unfinished cycle: committing only happens at genuine cycle
	// boundaries, so segmented runs are arithmetically identical to a
	// one-shot solve.
	rel := s.boundaryRitual(A, b, s.j >= s.m)
	if rel <= s.cfg.Tolerance {
		return StatusConverged
	}
	if atTotalCap {
		return StatusMaxIterations
	}
	return StatusPaused
}

func (s *gmresState) checkCompatible(cfg Config, n int) error {
	if s.n != n {
		return ErrProgressMismatch
	}
	if cfg.Method != MethodGMRES || !s.cfg.match(cfg) {
		return ErrProgressMismatch
	}
	return nil
}

func safeScale(resid, bNorm float64) float64 {
	if bNorm == 0 {
		return resid
	}
	return resid / bNorm
}

// mulV forms the linear combination sum_{i<k} y[i] V[i].
func mulV(V [][]float64, y []float64, k int) []float64 {
	out := make([]float64, len(V[0]))
	for i := 0; i < k; i++ {
		yi := y[i]
		vi := V[i]
		for r := range out {
			out[r] += yi * vi[r]
		}
	}
	return out
}

// backSolve solves R y = g for the leading k x k upper-triangular R,
// stored column-major with stride m (Givens rotations already applied).
func backSolve(H, g []float64, k, m int) []float64 {
	y := make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		su := g[i]
		for jj := i + 1; jj < k; jj++ {
			su -= H[i*m+jj] * y[jj]
		}
		d := H[i*m+i]
		if d == 0 {
			y[i] = 0
		} else {
			y[i] = su / d
		}
	}
	return y
}

func axpyIntoX(x, add []float64) []float64 {
	for i := range x {
		x[i] += add[i]
	}
	return x
}

func scale(a float64, x []float64) []float64 {
	y := make([]float64, len(x))
	for i := range x {
		y[i] = a * x[i]
	}
	return y
}

func axpy(a float64, x, y []float64) []float64 {
	z := make([]float64, len(y))
	for i := range y {
		z[i] = a*x[i] + y[i]
	}
	return z
}

// clone returns a deep copy of the checkpoint, including an unfinished
// Arnoldi cycle, so an archived Progress keeps its exact measured state.
func (s *gmresState) clone() *gmresState {
	c := *s
	c.x = cloneFloats(s.x)
	c.r = cloneFloats(s.r)
	c.diag = cloneFloats(s.diag)
	c.history = append([]float64(nil), s.history...)
	if s.V != nil {
		c.V = make([][]float64, len(s.V))
		for i := range s.V {
			if s.V[i] != nil {
				c.V[i] = cloneFloats(s.V[i])
			}
		}
		c.H = cloneFloats(s.H)
		c.cs = cloneFloats(s.cs)
		c.sn = cloneFloats(s.sn)
		c.g = cloneFloats(s.g)
	}
	return &c
}
