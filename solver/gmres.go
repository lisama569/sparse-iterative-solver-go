package solver

import (
	"context"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// solveGMRES runs restarted GMRES(m) with modified Gram-Schmidt Arnoldi
// and Givens-rotation least squares. It makes no symmetry assumption and
// therefore applies to general (non-symmetric) square systems.
//
// Optional Jacobi right preconditioning solves (A M^-1) y = b and returns
// x = M^-1 y.
func solveGMRES(ctx context.Context, A *sparse.CSR, b, x []float64, cfg Config) *Result {
	n := len(x)
	m := cfg.Restart
	if m > n {
		m = n
	}
	if m <= 0 {
		return &Result{Status: StatusInvalidInput, X: x, ErrorMessage: "GMRES restart length must be positive"}
	}
	res := &Result{Status: StatusMaxIterations, X: x}

	var diag []float64
	if cfg.Precondition {
		diag = A.Diagonal()
	}
	applyMinv := func(v []float64) []float64 {
		out := make([]float64, n)
		if cfg.Precondition {
			for i := 0; i < n; i++ {
				out[i] = v[i] / diag[i]
			}
		} else {
			copy(out, v)
		}
		return out
	}
	// amp computes w = A M^-1 v.
	amp := func(v []float64) ([]float64, error) {
		return A.MatVec(applyMinv(v))
	}

	bNorm := vnorm(b)
	total := 0

	// Initial residual.
	Ax, err := A.MatVec(x)
	if err != nil {
		return invalidFrom(err, x)
	}
	r := make([]float64, n)
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

	for total < cfg.MaxIterations {
		beta := vnorm(r)
		if beta == 0 || math.IsNaN(beta) {
			res.Status = StatusBreakdown
			res.ErrorMessage = "GMRES residual norm not usable: " + floatStr(beta)
			res.FinalRelRes = rel
			res.Iterations = total
			return res
		}

		// V holds m+1 Krylov vectors; H is upper Hessenberg (m+1 x m)
		// stored column-major with stride m.
		V := make([][]float64, m+1)
		V[0] = scale(1/beta, r)
		H := make([]float64, (m+1)*m)
		cs := make([]float64, m)
		sn := make([]float64, m)
		g := make([]float64, m+1)
		g[0] = beta

		j := 0
		for ; j < m && total < cfg.MaxIterations; j++ {
			select {
			case <-ctx.Done():
				res.Status = StatusCanceled
				res.ErrorMessage = ctx.Err().Error()
				res.FinalRelRes = rel
				res.Iterations = total
				return res
			default:
			}

			w, err := amp(V[j])
			if err != nil {
				return invalidFrom(err, x)
			}
			if !finiteVec(w) {
				res.Status = StatusBreakdown
				res.ErrorMessage = "non-finite Arnoldi vector"
				res.FinalRelRes = math.NaN()
				res.Iterations = total
				return res
			}
			// Modified Gram-Schmidt.
			for i := 0; i <= j; i++ {
				hij := vdot(w, V[i])
				H[i*m+j] = hij
				w = axpy(-hij, V[i], w)
			}
			hnorm := vnorm(w)
			H[(j+1)*m+j] = hnorm

			// Apply previously computed Givens rotations G_i to column j.
			// The two affected rows transform as
			// (a,b) -> (c*a + s*b, -s*a + c*b).
			for i := 0; i < j; i++ {
				h1 := H[i*m+j]
				h2 := H[(i+1)*m+j]
				H[i*m+j] = cs[i]*h1 + sn[i]*h2
				H[(i+1)*m+j] = -sn[i]*h1 + cs[i]*h2
			}
			h1 := H[j*m+j]
			h2 := H[(j+1)*m+j]
			denom := math.Hypot(h1, h2)
			if denom == 0 || math.IsNaN(denom) {
				// Lucky breakdown: A-invariant Krylov subspace. The least
				// squares residual |g[j+1]| (which stays at |g[j]| since
				// the rotation is undefined) tells us whether the solve
				// actually succeeded.
				resid := math.Abs(g[j])
				rel = safeScale(resid, bNorm)
				if rel <= cfg.Tolerance {
					y := backSolve(H, g, j, m)
					x = axpyIntoX(x, applyMinv(mulV(V, y, j)))
					res.X = x
					res.Status = StatusConverged
					res.FinalRelRes = rel
					res.Iterations = total
					res.History = append(res.History, rel)
					return res
				}
				res.Status = StatusBreakdown
				res.ErrorMessage = "GMRES breakdown: zero Krylov basis norm without convergence"
				res.FinalRelRes = rel
				res.Iterations = total
				return res
			}
			// New rotation G_j with c = h1/d, s = h2/d zeroes H[j+1,j]:
			// (h1,h2) -> (c*h1+s*h2, -s*h1+c*h2) = (d,0).
			cs[j] = h1 / denom
			sn[j] = h2 / denom
			H[j*m+j] = denom
			H[(j+1)*m+j] = 0
			gj := g[j]
			g[j] = cs[j]*gj + sn[j]*g[j+1]
			g[j+1] = -sn[j]*gj + cs[j]*g[j+1]

			total++
			rel = safeScale(math.Abs(g[j+1]), bNorm)
			res.History = append(res.History, rel)
			if rel <= cfg.Tolerance {
				y := backSolve(H, g, j+1, m)
				x = axpyIntoX(x, applyMinv(mulV(V, y, j+1)))
				res.X = x
				res.Status = StatusConverged
				res.FinalRelRes = rel
				res.Iterations = total
				return res
			}
			if hnorm == 0 {
				// Invariant subspace reached without convergence: more
				// restarts cannot change the result.
				res.Status = StatusBreakdown
				res.ErrorMessage = "GMRES breakdown: invariant Krylov subspace before convergence"
				res.FinalRelRes = rel
				res.Iterations = total
				return res
			}
			V[j+1] = scale(1/hnorm, w)
		}

		// Restart: update x with the best approximation so far and
		// recompute the true residual explicitly (never trust only the
		// projected residual across a restart).
		y := backSolve(H, g, j, m)
		x = axpyIntoX(x, applyMinv(mulV(V, y, j)))
		res.X = x
		Ax, err = A.MatVec(x)
		if err != nil {
			return invalidFrom(err, x)
		}
		for i := 0; i < n; i++ {
			r[i] = b[i] - Ax[i]
		}
		rel = safeRelNorm(r, bNorm)
		res.History = append(res.History, rel)
		if rel <= cfg.Tolerance {
			res.Status = StatusConverged
			res.FinalRelRes = rel
			res.Iterations = total
			return res
		}
	}
	// Budget exhausted: report an explicitly measured residual.
	Ax, _ = A.MatVec(x)
	for i := 0; i < n; i++ {
		r[i] = b[i] - Ax[i]
	}
	res.X = x
	res.FinalRelRes = safeRelNorm(r, bNorm)
	res.History = append(res.History, res.FinalRelRes)
	res.Iterations = total
	if res.FinalRelRes <= cfg.Tolerance {
		res.Status = StatusConverged
	}
	return res
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
		s := g[i]
		for jj := i + 1; jj < k; jj++ {
			s -= H[i*m+jj] * y[jj]
		}
		d := H[i*m+i]
		if d == 0 {
			y[i] = 0
		} else {
			y[i] = s / d
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
