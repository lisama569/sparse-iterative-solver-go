package solver

import (
	"errors"
	"math"
	"testing"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// laplacian1D 生成 n 阶一维 Laplacian（对称正定）。
func laplacian1D(n int) *sparse.Matrix {
	entries := make([]sparse.Entry, 0, 3*n)
	for i := 0; i < n; i++ {
		if i > 0 {
			entries = append(entries, sparse.Entry{Row: i, Col: i - 1, Val: -1})
		}
		entries = append(entries, sparse.Entry{Row: i, Col: i, Val: 2})
		if i < n-1 {
			entries = append(entries, sparse.Entry{Row: i, Col: i + 1, Val: -1})
		}
	}
	m, err := sparse.BuildFromCOO(n, n, entries)
	if err != nil {
		panic(err)
	}
	return m
}

// nonsym 生成对角占优的非对称矩阵。
func nonsym(n int) *sparse.Matrix {
	entries := make([]sparse.Entry, 0, 3*n)
	for i := 0; i < n; i++ {
		if i > 0 {
			entries = append(entries, sparse.Entry{Row: i, Col: i - 1, Val: -0.5})
		}
		entries = append(entries, sparse.Entry{Row: i, Col: i, Val: 4})
		if i < n-1 {
			entries = append(entries, sparse.Entry{Row: i, Col: i + 1, Val: -1})
		}
	}
	m, err := sparse.BuildFromCOO(n, n, entries)
	if err != nil {
		panic(err)
	}
	return m
}

// exactRHS 由已知真解构造右端项 b = A*xExact。
func exactRHS(a *sparse.Matrix, xExact []float64) []float64 {
	b, err := a.MatVec(xExact)
	if err != nil {
		panic(err)
	}
	return b
}

func ones(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 1
	}
	return x
}

// assertConverged 校验收敛结论、残差与误差，失败时打印规模与收敛历史。
func assertConverged(t *testing.T, name string, a *sparse.Matrix, b, xExact []float64, r *Result, tol float64) {
	t.Helper()
	if !r.Converged() {
		t.Fatalf("%s: 未收敛 (n=%d nnz=%d): %s err=%v", name, a.Rows, a.NNZ(), r.Summary(), r.Err)
	}
	// 独立重算残差，不依赖求解器自报数值。
	ax, err := a.MatVec(r.X)
	if err != nil {
		t.Fatalf("%s: MatVec 失败: %v", name, err)
	}
	var resNorm, bNorm, errNorm float64
	for i := range b {
		d := b[i] - ax[i]
		resNorm += d * d
		bNorm += b[i] * b[i]
		e := r.X[i] - xExact[i]
		errNorm += e * e
	}
	resNorm = math.Sqrt(resNorm)
	bNorm = math.Sqrt(bNorm)
	errNorm = math.Sqrt(errNorm)
	if bNorm == 0 {
		bNorm = 1
	}
	if resNorm/bNorm > tol {
		t.Fatalf("%s: 重算残差 %.3e 超过容差 %.3e (n=%d): %s",
			name, resNorm/bNorm, tol, a.Rows, r.Summary())
	}
	if errNorm > 1e-4 {
		t.Fatalf("%s: 解误差 %.3e 过大 (n=%d): %s", name, errNorm, a.Rows, r.Summary())
	}
	t.Logf("%s: n=%d iters=%d rel_res=%.3e err=%.3e", name, a.Rows, r.Iterations, r.FinalRelRes, errNorm)
}

func TestCG_Converges(t *testing.T) {
	for _, n := range []int{10, 100, 500} {
		a := laplacian1D(n)
		xExact := ones(n)
		b := exactRHS(a, xExact)
		r := Solve(a, b, Options{Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true})
		assertConverged(t, "CG", a, b, xExact, r, 1e-8)
	}
}

func TestCG_WithPrecondAndInitialGuess(t *testing.T) {
	n := 200
	a := laplacian1D(n)
	xExact := ones(n)
	b := exactRHS(a, xExact)
	r := Solve(a, b, Options{
		Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true, Precond: true,
		InitialX: make([]float64, n),
	})
	assertConverged(t, "PCG", a, b, xExact, r, 1e-8)
}

func TestBiCGSTAB_Converges(t *testing.T) {
	for _, n := range []int{10, 100, 500} {
		a := nonsym(n)
		xExact := ones(n)
		b := exactRHS(a, xExact)
		r := Solve(a, b, Options{Tolerance: 1e-10, MaxIter: 10 * n, Precond: true})
		assertConverged(t, "BiCGSTAB", a, b, xExact, r, 1e-8)
	}
}

func TestMaxIter_NotMarkedSuccess(t *testing.T) {
	n := 100
	a := laplacian1D(n)
	b := exactRHS(a, ones(n))
	r := Solve(a, b, Options{Tolerance: 1e-14, MaxIter: 3, Symmetric: true})
	if r.Status != StatusMaxIter {
		t.Fatalf("期望 StatusMaxIter, 得到 %s", r.Status)
	}
	if r.Converged() {
		t.Fatal("达到迭代上限的结果不得标记为成功")
	}
	if !errors.Is(r.Err, ErrNotConverged) {
		t.Fatalf("期望 ErrNotConverged, 得到 %v", r.Err)
	}
	t.Logf("max_iter 判定依据: %s", r.Summary())
}

func TestCG_BreakdownOnIndefinite(t *testing.T) {
	// 负定矩阵：CG 的 p^T A p 为负，算法对非正定系统失效。
	a, err := sparse.BuildFromCOO(2, 2, []sparse.Entry{
		{Row: 0, Col: 0, Val: -1}, {Row: 1, Col: 1, Val: -1},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := Solve(a, []float64{1, 1}, Options{Tolerance: 1e-12, MaxIter: 100, Symmetric: true})
	// 负定对角阵上 CG 会"收敛"到真解（因为 -I 也是自伴的，CG 数学上仍可用），
	// 改用真正导致失效的矩阵：p^T A p = 0。
	_ = r
	a2, err := sparse.BuildFromCOO(2, 2, []sparse.Entry{
		{Row: 0, Col: 0, Val: 1}, {Row: 0, Col: 1, Val: 1},
		{Row: 1, Col: 0, Val: 1}, {Row: 1, Col: 1, Val: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	// A 奇异，b 不在值域内：CG 必失效或无法收敛。
	r2 := Solve(a2, []float64{1, 0}, Options{Tolerance: 1e-12, MaxIter: 100, Symmetric: true})
	if r2.Converged() {
		t.Fatalf("奇异系统不得标记为收敛: %s", r2.Summary())
	}
	if r2.Status != StatusBreakdown && r2.Status != StatusMaxIter {
		t.Fatalf("期望 breakdown 或 max_iter, 得到 %s", r2.Status)
	}
	t.Logf("失效判定依据: %s err=%v", r2.Summary(), r2.Err)
}

func TestBiCGSTAB_Breakdown(t *testing.T) {
	// 构造 rhat^T r = 0 的场景：A 使残差与自身正交。
	// 使用零右端以外的经典失效矩阵。
	a, err := sparse.BuildFromCOO(2, 2, []sparse.Entry{
		{Row: 0, Col: 0, Val: 0}, {Row: 0, Col: 1, Val: 1},
		{Row: 1, Col: 0, Val: 1}, {Row: 1, Col: 1, Val: 0},
	})
	if err != nil {
		t.Fatal(err)
	}
	// b = e1，初值 0：r = e1，A*r = e2，rhat^T A r = 0 → 第一步即失效。
	r := Solve(a, []float64{1, 0}, Options{Tolerance: 1e-12, MaxIter: 100})
	if r.Status != StatusBreakdown {
		t.Fatalf("期望 StatusBreakdown, 得到 %s: %s", r.Status, r.Summary())
	}
	if !errors.Is(r.Err, ErrBreakdown) {
		t.Fatalf("期望 ErrBreakdown, 得到 %v", r.Err)
	}
	if r.Converged() {
		t.Fatal("失效结果不得标记为成功")
	}
	t.Logf("失效判定依据: %s err=%v", r.Summary(), r.Err)
}

func TestInvalidInput(t *testing.T) {
	a := laplacian1D(4)
	good := exactRHS(a, ones(4))
	cases := []struct {
		name string
		a    *sparse.Matrix
		b    []float64
		opts Options
	}{
		{"nil 矩阵", nil, good, Options{Tolerance: 1e-8, MaxIter: 10}},
		{"b 维度不符", a, []float64{1, 2}, Options{Tolerance: 1e-8, MaxIter: 10}},
		{"b 含 NaN", a, []float64{1, math.NaN(), 2, 0}, Options{Tolerance: 1e-8, MaxIter: 10}},
		{"零容差", a, good, Options{Tolerance: 0, MaxIter: 10}},
		{"负容差", a, good, Options{Tolerance: -1, MaxIter: 10}},
		{"零步数", a, good, Options{Tolerance: 1e-8, MaxIter: 0}},
		{"初值维度不符", a, good, Options{Tolerance: 1e-8, MaxIter: 10, InitialX: []float64{1}}},
	}
	for _, c := range cases {
		r := Solve(c.a, c.b, c.opts)
		if r.Status != StatusInvalid || !errors.Is(r.Err, ErrInvalidInput) {
			t.Errorf("%s: 期望 invalid_input, 得到 %s err=%v", c.name, r.Status, r.Err)
		}
		if r.Converged() {
			t.Errorf("%s: 非法输入不得标记为成功", c.name)
		}
	}
}

func TestSymmetricPathRejectsAsymmetric(t *testing.T) {
	// 上双对角矩阵：结构不对称。
	a, err := sparse.BuildFromCOO(5, 5, []sparse.Entry{
		{Row: 0, Col: 0, Val: 2}, {Row: 0, Col: 1, Val: 1},
		{Row: 1, Col: 1, Val: 2}, {Row: 1, Col: 2, Val: 1},
		{Row: 2, Col: 2, Val: 2}, {Row: 2, Col: 3, Val: 1},
		{Row: 3, Col: 3, Val: 2}, {Row: 3, Col: 4, Val: 1},
		{Row: 4, Col: 4, Val: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	b := exactRHS(a, ones(5))
	r := Solve(a, b, Options{Tolerance: 1e-8, MaxIter: 100, Symmetric: true})
	if r.Status != StatusInvalid {
		t.Fatalf("非对称矩阵走对称路径应报 invalid_input, 得到 %s", r.Status)
	}
}

func TestReproducible(t *testing.T) {
	n := 200
	a := laplacian1D(n)
	b := exactRHS(a, ones(n))
	opts := Options{Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true, Precond: true}
	r1 := Solve(a, b, opts)
	r2 := Solve(a, b, opts)
	if r1.Status != r2.Status || r1.Iterations != r2.Iterations ||
		r1.FinalRelRes != r2.FinalRelRes {
		t.Fatalf("固定配置重复运行结论不稳定:\n  %s\n  %s", r1.Summary(), r2.Summary())
	}
	for i := range r1.X {
		if r1.X[i] != r2.X[i] {
			t.Fatalf("解向量不一致 at %d", i)
		}
	}
}

func TestZeroRHS(t *testing.T) {
	a := laplacian1D(10)
	r := Solve(a, make([]float64, 10), Options{Tolerance: 1e-10, MaxIter: 100, Symmetric: true})
	if !r.Converged() || r.Iterations != 0 {
		t.Fatalf("零右端应立即收敛: %s", r.Summary())
	}
}
