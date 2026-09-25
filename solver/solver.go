// Package solver 提供面向对称与一般稀疏系统的迭代求解与并发执行。
package solver

import (
	"errors"
	"fmt"
	"math"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// 求解相关错误分类。
var (
	ErrInvalidInput = errors.New("solver: 输入不合法")
	ErrNotConverged = errors.New("solver: 达到迭代上限未收敛")
	ErrBreakdown    = errors.New("solver: 算法失效")
	ErrCanceled     = errors.New("solver: 任务被取消")
)

// Status 区分求解结论。
type Status int

const (
	StatusConverged Status = iota // 残差满足容差
	StatusMaxIter                 // 达到迭代上限
	StatusBreakdown               // 算法失效（如 rho=0）
	StatusInvalid                 // 输入不合法
	StatusCanceled                // 被取消
)

func (s Status) String() string {
	switch s {
	case StatusConverged:
		return "converged"
	case StatusMaxIter:
		return "max_iter"
	case StatusBreakdown:
		return "breakdown"
	case StatusInvalid:
		return "invalid_input"
	case StatusCanceled:
		return "canceled"
	}
	return "unknown"
}

// Options 控制一次求解。
type Options struct {
	Tolerance float64   // 相对残差容差，必须 > 0 且有限
	MaxIter   int       // 最大步数，必须 > 0
	InitialX  []float64 // 可选初值，nil 表示零向量
	Symmetric  bool     // true 走 CG，false 走 BiCGSTAB
	Precond    bool     // 是否启用 Jacobi 预条件
}

// Result 记录求解结论与可解释的判定依据。
type Result struct {
	Status      Status
	X           []float64
	Iterations  int
	FinalRelRes float64   // 最终相对残差 ||r||/||b||
	ResHistory  []float64 // 每步相对残差
	Err         error
}

// Converged 报告是否真正满足残差要求。
func (r *Result) Converged() bool { return r.Status == StatusConverged }

// Summary 返回收敛历史摘要，用于失败日志。
func (r *Result) Summary() string {
	hist := r.ResHistory
	const keep = 5
	head := hist
	if len(head) > keep {
		head = head[:keep]
	}
	tail := ""
	if len(hist) > keep {
		tail = fmt.Sprintf(" ... 末段 %v", hist[len(hist)-keep:])
	}
	return fmt.Sprintf("status=%s iters=%d final_rel_res=%.3e history=%v%s",
		r.Status, r.Iterations, r.FinalRelRes, head, tail)
}

// Solve 求解 A*x = b。任何情况下未满足残差要求都不会标记为成功。
func Solve(a *sparse.Matrix, b []float64, opts Options) *Result {
	if err := validate(a, b, opts); err != nil {
		return &Result{Status: StatusInvalid, Err: err, FinalRelRes: math.Inf(1)}
	}
	if opts.Symmetric {
		return cg(a, b, opts)
	}
	return bicgstab(a, b, opts)
}

// validate 校验矩阵、右端项与选项；返回可区分的输入错误。
func validate(a *sparse.Matrix, b []float64, opts Options) error {
	if a == nil {
		return fmt.Errorf("%w: 矩阵为 nil", ErrInvalidInput)
	}
	if a.Rows != a.Cols {
		return fmt.Errorf("%w: %w: %d x %d", ErrInvalidInput, sparse.ErrNotSquare, a.Rows, a.Cols)
	}
	if len(b) != a.Rows {
		return fmt.Errorf("%w: %w: A 是 %d x %d, b 长度 %d",
			ErrInvalidInput, sparse.ErrDimension, a.Rows, a.Cols, len(b))
	}
	for i, v := range b {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%w: %w: b[%d]=%v", ErrInvalidInput, sparse.ErrBadValue, i, v)
		}
	}
	if !(opts.Tolerance > 0) || math.IsNaN(opts.Tolerance) || math.IsInf(opts.Tolerance, 0) {
		return fmt.Errorf("%w: 容差必须为正的有限值, got %v", ErrInvalidInput, opts.Tolerance)
	}
	if opts.MaxIter <= 0 {
		return fmt.Errorf("%w: 最大步数必须为正, got %d", ErrInvalidInput, opts.MaxIter)
	}
	if opts.InitialX != nil && len(opts.InitialX) != a.Cols {
		return fmt.Errorf("%w: %w: 初值长度 %d, 需要 %d",
			ErrInvalidInput, sparse.ErrDimension, len(opts.InitialX), a.Cols)
	}
	if opts.Symmetric {
		if err := a.CheckSymmetricStructure(); err != nil {
			return fmt.Errorf("%w: 对称路径要求结构对称: %v", ErrInvalidInput, err)
		}
	}
	return nil
}

// --- 内部工具 ---

func dot(x, y []float64) float64 {
	var s float64
	for i := range x {
		s += x[i] * y[i]
	}
	return s
}

func norm2(x []float64) float64 { return math.Sqrt(dot(x, x)) }

// mustMatVec 包装 MatVec；Solve 已校验维度，这里不应失败。
func mustMatVec(a *sparse.Matrix, x []float64) []float64 {
	y, err := a.MatVec(x)
	if err != nil {
		panic(fmt.Sprintf("solver: 内部维度错误: %v", err))
	}
	return y
}

// newPrecond 按需构建 Jacobi 预条件子。
func newPrecond(a *sparse.Matrix, opts Options) (*jacobi, error) {
	if !opts.Precond {
		return nil, nil
	}
	j, err := newJacobi(a)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidInput, err)
	}
	return j, nil
}

