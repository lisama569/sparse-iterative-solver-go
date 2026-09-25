package solver

import (
	"fmt"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// jacobi 是 Jacobi（对角）预条件子，M = diag(A)。
type jacobi struct {
	invDiag []float64
}

// newJacobi 构建预条件子；对角元缺失或为零时返回错误。
func newJacobi(a *sparse.Matrix) (*jacobi, error) {
	d, err := a.Diagonal()
	if err != nil {
		return nil, fmt.Errorf("solver: 无法构建 Jacobi 预条件: %w", err)
	}
	inv := make([]float64, len(d))
	for i, v := range d {
		inv[i] = 1.0 / v
	}
	return &jacobi{invDiag: inv}, nil
}

// apply 计算 z = M^{-1} r。
func (j *jacobi) apply(r []float64) []float64 {
	z := make([]float64, len(r))
	for i := range r {
		z[i] = j.invDiag[i] * r[i]
	}
	return z
}
