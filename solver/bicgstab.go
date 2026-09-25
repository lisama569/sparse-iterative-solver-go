package solver

import (
	"fmt"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// bicgstab 是（可选 Jacobi 预条件的）BiCGSTAB，适用于一般非对称系统。
func bicgstab(a *sparse.Matrix, b []float64, opts Options) *Result {
	pc, err := newPrecond(a, opts)
	if err != nil {
		return &Result{Status: StatusInvalid, Err: err, FinalRelRes: math.Inf(1)}
	}

	n := a.Rows
	x := make([]float64, n)
	if opts.InitialX != nil {
		copy(x, opts.InitialX)
	}
	ax := mustMatVec(a, x)
	r := make([]float64, n)
	for i := range r {
		r[i] = b[i] - ax[i]
	}
	bnorm := norm2(b)
	if bnorm == 0 {
		bnorm = 1
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

	rhat := make([]float64, n)
	copy(rhat, r)
	p := make([]float64, n)
	v := make([]float64, n)
	rho, alpha, omega := 1.0, 1.0, 1.0

	breakdown := func(k int, msg string) *Result {
		res.Status = StatusBreakdown
		res.Err = fmt.Errorf("%w: BiCGSTAB 第 %d 步: %s", ErrBreakdown, k, msg)
		res.Iterations = k
		res.FinalRelRes = res.ResHistory[len(res.ResHistory)-1]
		return res
	}

	for k := 1; k <= opts.MaxIter; k++ {
		rhoNew := dot(rhat, r)
		if rhoNew == 0 || math.IsNaN(rhoNew) {
			return breakdown(k, fmt.Sprintf("rho = %v", rhoNew))
		}
		beta := (rhoNew / rho) * (alpha / omega)
		for i := 0; i < n; i++ {
			p[i] = r[i] + beta*(p[i]-omega*v[i])
		}
		phat := applyPC(p)
		v = mustMatVec(a, phat)
		den := dot(rhat, v)
		if den == 0 || math.IsNaN(den) {
			return breakdown(k, fmt.Sprintf("rhat^T v = %v", den))
		}
		alpha = rhoNew / den
		s := make([]float64, n)
		for i := 0; i < n; i++ {
			s[i] = r[i] - alpha*v[i]
		}
		// 提前收敛检查（s 已足够小）。
		snorm := norm2(s) / bnorm
		if snorm <= opts.Tolerance {
			for i := 0; i < n; i++ {
				x[i] += alpha * phat[i]
			}
			res.ResHistory = append(res.ResHistory, snorm)
			res.Status = StatusConverged
			res.Iterations = k
			res.FinalRelRes = snorm
			return res
		}
		shat := applyPC(s)
		t := mustMatVec(a, shat)
		tt := dot(t, t)
		if tt == 0 || math.IsNaN(tt) {
			return breakdown(k, fmt.Sprintf("t^T t = %v", tt))
		}
		omega = dot(t, s) / tt
		if omega == 0 || math.IsNaN(omega) {
			return breakdown(k, fmt.Sprintf("omega = %v", omega))
		}
		for i := 0; i < n; i++ {
			x[i] += alpha*phat[i] + omega*shat[i]
			r[i] = s[i] - omega*t[i]
		}
		relres = norm2(r) / bnorm
		res.ResHistory = append(res.ResHistory, relres)
		if math.IsNaN(relres) || math.IsInf(relres, 0) {
			return breakdown(k, fmt.Sprintf("残差发散: %v", relres))
		}
		if relres <= opts.Tolerance {
			res.Status = StatusConverged
			res.Iterations = k
			res.FinalRelRes = relres
			return res
		}
		rho = rhoNew
	}
	res.Status = StatusMaxIter
	res.Err = fmt.Errorf("%w: %d 步后相对残差 %.3e > 容差 %.3e",
		ErrNotConverged, opts.MaxIter, relres, opts.Tolerance)
	res.Iterations = opts.MaxIter
	res.FinalRelRes = relres
	return res
}
