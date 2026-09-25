package solver

import (
	"fmt"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// cg 是（可选 Jacobi 预条件的）共轭梯度法，适用于对称正定系统。
func cg(a *sparse.Matrix, b []float64, opts Options) *Result {
	pc, err := newPrecond(a, opts)
	if err != nil {
		return &Result{Status: StatusInvalid, Err: err, FinalRelRes: math.Inf(1)}
	}

	n := a.Rows
	x := make([]float64, n)
	if opts.InitialX != nil {
		copy(x, opts.InitialX)
	}
	// r = b - A*x
	ax := mustMatVec(a, x)
	r := make([]float64, n)
	for i := range r {
		r[i] = b[i] - ax[i]
	}
	bnorm := norm2(b)
	if bnorm == 0 {
		bnorm = 1 // b=0 时用绝对残差判据
	}

	res := &Result{X: x}
	relres := norm2(r) / bnorm
	res.ResHistory = append(res.ResHistory, relres)
	if relres <= opts.Tolerance {
		res.Status = StatusConverged
		res.FinalRelRes = relres
		return res
	}

	applyPC := func(v []float64) []float64 {
		if pc != nil {
			return pc.apply(v)
		}
		out := make([]float64, len(v))
		copy(out, v)
		return out
	}

	z := applyPC(r)
	p := make([]float64, n)
	copy(p, z)
	rz := dot(r, z)

	for k := 1; k <= opts.MaxIter; k++ {
		ap := mustMatVec(a, p)
		den := dot(p, ap)
		if den == 0 || math.IsNaN(den) || math.IsInf(den, 0) {
			res.Status = StatusBreakdown
			res.Err = fmt.Errorf("%w: CG 第 %d 步 p^T A p = %v（矩阵可能非正定）", ErrBreakdown, k, den)
			res.Iterations = k - 1
			res.FinalRelRes = res.ResHistory[len(res.ResHistory)-1]
			return res
		}
		alpha := rz / den
		for i := 0; i < n; i++ {
			x[i] += alpha * p[i]
			r[i] -= alpha * ap[i]
		}
		relres = norm2(r) / bnorm
		res.ResHistory = append(res.ResHistory, relres)
		if math.IsNaN(relres) || math.IsInf(relres, 0) {
			res.Status = StatusBreakdown
			res.Err = fmt.Errorf("%w: CG 第 %d 步残差发散: %v", ErrBreakdown, k, relres)
			res.Iterations = k
			res.FinalRelRes = relres
			return res
		}
		if relres <= opts.Tolerance {
			res.Status = StatusConverged
			res.Iterations = k
			res.FinalRelRes = relres
			return res
		}
		z = applyPC(r)
		rzNew := dot(r, z)
		if rzNew == 0 {
			res.Status = StatusBreakdown
			res.Err = fmt.Errorf("%w: CG 第 %d 步 r^T z = 0", ErrBreakdown, k)
			res.Iterations = k
			res.FinalRelRes = relres
			return res
		}
		beta := rzNew / rz
		for i := 0; i < n; i++ {
			p[i] = z[i] + beta*p[i]
		}
		rz = rzNew
	}
	res.Status = StatusMaxIter
	res.Err = fmt.Errorf("%w: %d 步后相对残差 %.3e > 容差 %.3e",
		ErrNotConverged, opts.MaxIter, relres, opts.Tolerance)
	res.Iterations = opts.MaxIter
	res.FinalRelRes = relres
	return res
}
