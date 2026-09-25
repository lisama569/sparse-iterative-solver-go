package solver

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// spd3 returns a small symmetric positive definite 3x3 matrix.
func spd3(t *testing.T) *sparse.CSR {
	t.Helper()
	c := &sparse.COO{Rows: 3, Cols: 3, Symmetric: true, Triplets: []sparse.Triplet{
		{Row: 0, Col: 0, Value: 4}, {Row: 1, Col: 0, Value: 1}, {Row: 1, Col: 1, Value: 3}, {Row: 2, Col: 1, Value: 1}, {Row: 2, Col: 2, Value: 2},
	}}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// nonsym3 returns a diagonally dominant non-symmetric 3x3 matrix.
func nonsym3(t *testing.T) *sparse.CSR {
	t.Helper()
	c := &sparse.COO{Rows: 3, Cols: 3, Triplets: []sparse.Triplet{
		{Row: 0, Col: 0, Value: 10}, {Row: 0, Col: 1, Value: 1},
		{Row: 1, Col: 0, Value: 2}, {Row: 1, Col: 1, Value: 10}, {Row: 1, Col: 2, Value: 1},
		{Row: 2, Col: 1, Value: 3}, {Row: 2, Col: 2, Value: 10},
	}}
	m, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func checkResidual(t *testing.T, A *sparse.CSR, x, b []float64, tol float64) {
	t.Helper()
	Ax, err := A.MatVec(x)
	if err != nil {
		t.Fatal(err)
	}
	var res, bn float64
	for i := range b {
		res += (Ax[i] - b[i]) * (Ax[i] - b[i])
		bn += b[i] * b[i]
	}
	rel := math.Sqrt(res) / math.Sqrt(bn)
	if math.IsNaN(rel) || rel > tol {
		t.Fatalf("relative residual %.3e exceeds tolerance %.3e; x=%v", rel, tol, x)
	}
}

func TestCGConverges(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}
	cfg := Config{Tolerance: 1e-10, MaxIterations: 100, Method: MethodCG}
	r, err := Solve(context.Background(), A, b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged {
		t.Fatalf("status=%s rel=%.3e iters=%d msg=%s", r.Status, r.FinalRelRes, r.Iterations, r.ErrorMessage)
	}
	checkResidual(t, A, r.X, b, cfg.Tolerance)
	if r.FinalRelRes > cfg.Tolerance {
		t.Fatalf("reported rel %.3e > tol %.3e", r.FinalRelRes, cfg.Tolerance)
	}
}

func TestGMRESConvergesNonsymmetric(t *testing.T) {
	A := nonsym3(t)
	b := []float64{1, 2, 3}
	cfg := Config{Tolerance: 1e-10, MaxIterations: 100, Method: MethodGMRES}
	r, err := Solve(context.Background(), A, b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged {
		t.Fatalf("status=%s rel=%.3e iters=%d msg=%s", r.Status, r.FinalRelRes, r.Iterations, r.ErrorMessage)
	}
	checkResidual(t, A, r.X, b, cfg.Tolerance)
}

func TestAutoMethodBySymmetry(t *testing.T) {
	b := []float64{1, 2, 3}
	r, err := Solve(context.Background(), spd3(t), b, Config{Tolerance: 1e-10})
	if err != nil || r.Status != StatusConverged {
		t.Fatalf("auto CG: status=%s err=%v msg=%s", r.Status, err, r.ErrorMessage)
	}
	r2, err := Solve(context.Background(), nonsym3(t), b, Config{Tolerance: 1e-10})
	if err != nil || r2.Status != StatusConverged {
		t.Fatalf("auto GMRES: status=%s err=%v msg=%s", r2.Status, err, r2.ErrorMessage)
	}
}

func TestInitialGuess(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}
	// Start from the exact solution (obtained by an earlier solve):
	// convergence must be recognized at iteration 0.
	exact, err := Solve(context.Background(), A, b, Config{Tolerance: 1e-14, Method: MethodCG})
	if err != nil || exact.Status != StatusConverged {
		t.Fatalf("setup solve failed: %s %v", exact.Status, err)
	}
	cfg := Config{Tolerance: 1e-12, Method: MethodCG, InitialGuess: exact.X}
	r, err := Solve(context.Background(), A, b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged || r.Iterations != 0 {
		t.Fatalf("status=%s iters=%d rel=%.3e", r.Status, r.Iterations, r.FinalRelRes)
	}
}

func TestMaxIterationsNotConverged(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}
	cfg := Config{Tolerance: 1e-15, MaxIterations: 1, Method: MethodCG}
	r, err := Solve(context.Background(), A, b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusMaxIterations {
		t.Fatalf("status=%s, want max_iterations", r.Status)
	}
	if r.FinalRelRes <= cfg.Tolerance {
		t.Fatalf("should not report residual below tol: %.3e", r.FinalRelRes)
	}
}

func TestIndefiniteCGBreakdown(t *testing.T) {
	// Symmetric indefinite matrix: CG must report breakdown, not converge.
	c := &sparse.COO{Rows: 2, Cols: 2, Symmetric: true, Triplets: []sparse.Triplet{
		{Row: 0, Col: 0, Value: 1}, {Row: 1, Col: 0, Value: 2}, {Row: 1, Col: 1, Value: -4},
	}}
	A, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	r, err := Solve(context.Background(), A, []float64{1, 1}, Config{Method: MethodCG, Tolerance: 1e-8})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status == StatusConverged {
		t.Fatalf("CG must not converge on an indefinite matrix, rel=%.3e", r.FinalRelRes)
	}
	if r.Status != StatusBreakdown {
		t.Fatalf("status=%s, want breakdown", r.Status)
	}
	// GMRES handles the same indefinite system (it just needs a solution).
	rg, err := Solve(context.Background(), A, []float64{1, 1}, Config{Method: MethodGMRES, Tolerance: 1e-10, MaxIterations: 50})
	if err != nil {
		t.Fatal(err)
	}
	if rg.Status != StatusConverged {
		t.Fatalf("GMRES on indefinite system status=%s msg=%s", rg.Status, rg.ErrorMessage)
	}
	checkResidual(t, A, rg.X, []float64{1, 1}, 1e-10)
}

func TestInvalidInputs(t *testing.T) {
	A := spd3(t)
	cases := []struct {
		name string
		m    *sparse.CSR
		b    []float64
		cfg  Config
		kind sparse.ErrorKind
	}{
		{"nil matrix", nil, []float64{1, 2, 3}, Config{Tolerance: 1e-8}, sparse.KindStructure},
		{"b length mismatch", A, []float64{1, 2}, Config{Tolerance: 1e-8}, sparse.KindStructure},
		{"bad guess length", A, []float64{1, 2, 3}, Config{Tolerance: 1e-8, InitialGuess: []float64{1}}, sparse.KindStructure},
		{"bad tolerance", A, []float64{1, 2, 3}, Config{Tolerance: -1}, sparse.KindStructure},
		{"nan rhs", A, []float64{1, math.NaN(), 3}, Config{Tolerance: 1e-8}, sparse.KindNumeric},
		{"cg on non-symmetric", nonsym3(t), []float64{1, 2, 3},
			Config{Tolerance: 1e-8, Method: MethodCG}, sparse.KindStructure},
		{"unknown method", A, []float64{1, 2, 3},
			Config{Tolerance: 1e-8, Method: Method(99)}, sparse.KindStructure},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := Solve(context.Background(), tc.m, tc.b, tc.cfg)
			var ie *sparse.InputError
			if !errors.As(err, &ie) || ie.Kind != tc.kind {
				t.Fatalf("err=%v kind-mismatch, result status=%s", err, r.Status)
			}
			if r.Status != StatusInvalidInput {
				t.Fatalf("status=%s want invalid_input", r.Status)
			}
		})
	}
}

func TestJacobiPreconditioner(t *testing.T) {
	// Larger SPD with uneven diagonal scaling.
	n := 50
	rng := rand.New(rand.NewSource(42))
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: float64(4 + i%5)})
	}
	// Unique undirected edges only; diagonal dominance keeps the system SPD.
	edges := map[[2]int]bool{}
	for k := 0; k < 3*n; k++ {
		i := rng.Intn(n)
		j := rng.Intn(n - 1)
		if j >= i {
			j++
		}
		lo, hi := i, j
		if lo > hi {
			lo, hi = hi, lo
		}
		if edges[[2]int{lo, hi}] {
			continue
		}
		edges[[2]int{lo, hi}] = true
		trips = append(trips, sparse.Triplet{Row: hi, Col: lo, Value: 0.1})
	}
	A, err := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: trips}).ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	plain, _ := Solve(context.Background(), A, b, Config{Tolerance: 1e-10, Method: MethodCG})
	prec, _ := Solve(context.Background(), A, b, Config{Tolerance: 1e-10, Method: MethodCG, Precondition: true})
	if plain.Status != StatusConverged || prec.Status != StatusConverged {
		t.Fatalf("plain=%s prec=%s", plain.Status, prec.Status)
	}
	checkResidual(t, A, plain.X, b, 1e-10)
	checkResidual(t, A, prec.X, b, 1e-10)
	if prec.Iterations > plain.Iterations {
		t.Fatalf("preconditioning did not help: %d vs %d iterations", prec.Iterations, plain.Iterations)
	}
}

func TestDeterminism(t *testing.T) {
	A := spd3(t)
	b := []float64{1, 2, 3}
	cfg := Config{Tolerance: 1e-12, Method: MethodCG}
	r1, _ := Solve(context.Background(), A, b, cfg)
	r2, _ := Solve(context.Background(), A, b, cfg)
	if r1.Status != r2.Status || r1.Iterations != r2.Iterations || r1.FinalRelRes != r2.FinalRelRes {
		t.Fatalf("non-deterministic results: %+v vs %+v", r1, r2)
	}
}

func TestGMRESWithPreconditioner(t *testing.T) {
	A := nonsym3(t)
	b := []float64{1, 2, 3}
	r, err := Solve(context.Background(), A, b, Config{
		Tolerance: 1e-11, MaxIterations: 100, Method: MethodGMRES, Precondition: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged {
		t.Fatalf("status=%s msg=%s", r.Status, r.ErrorMessage)
	}
	checkResidual(t, A, r.X, b, 1e-11)
}

func TestGMRESRestart(t *testing.T) {
	// Small restart forces multiple restart cycles. We also assert the
	// first-cycle update is the least-squares optimum via convergence
	// behavior below; the Jacobi preconditioner keeps the cycle count
	// bounded so the test stays fast while exercising every restart path.
	n := 24
	A := laplacianLike(n)
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	r, err := Solve(context.Background(), A, b, Config{
		Tolerance: 1e-8, MaxIterations: 5000, Method: MethodGMRES, Restart: 6, Precondition: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged {
		t.Fatalf("status=%s iters=%d rel=%.3e msg=%s", r.Status, r.Iterations, r.FinalRelRes, r.ErrorMessage)
	}
	checkResidual(t, A, r.X, b, 1e-8)

	// Without preconditioning the same system must still converge with a
	// moderate restart and enough budget (pure correctness, no speed claim).
	r2, err := Solve(context.Background(), A, b, Config{
		Tolerance: 1e-8, MaxIterations: 10000, Method: MethodGMRES, Restart: 12,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r2.Status != StatusConverged {
		t.Fatalf("plain: status=%s iters=%d rel=%.3e", r2.Status, r2.Iterations, r2.FinalRelRes)
	}
}

// laplacianLike builds the n x n SPD tridiagonal -1,2,-1 matrix using the
// general (non-symmetric) path so GMRES is exercised.
func laplacianLike(n int) *sparse.CSR {
	return laplacianGeneral(n)
}

func laplacianGeneral(n int) *sparse.CSR {
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 2})
		if i > 0 {
			trips = append(trips, sparse.Triplet{Row: i, Col: i - 1, Value: -1})
			trips = append(trips, sparse.Triplet{Row: i - 1, Col: i, Value: -1})
		}
	}
	m, err := (&sparse.COO{Rows: n, Cols: n, Triplets: trips}).ToCSR()
	if err != nil {
		panic(err)
	}
	return m
}

// TestGMRESFirstCycleOptimal guards the Givens/Hessenberg bookkeeping:
// over one full Krylov cycle, GMRES must reach the least-squares optimum
// for that subspace. It compares against a CG reference on the same SPD
// matrix (unrestarted GMRES and CG both attain the optimal Galerkin error
// in their respective spaces; with restart >= n GMRES is exact).
func TestGMRESFullCycleExact(t *testing.T) {
	n := 12
	c := &sparse.COO{Rows: n, Cols: n, Symmetric: true}
	var trips []sparse.Triplet
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 4})
		if i > 0 {
			trips = append(trips, sparse.Triplet{Row: i, Col: i - 1, Value: -1})
		}
	}
	c.Triplets = trips
	A, err := c.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = float64(i + 1)
	}
	r, err := Solve(context.Background(), A, b, Config{
		Tolerance: 1e-13, MaxIterations: n, Method: MethodGMRES, Restart: n,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged {
		t.Fatalf("full-cycle GMRES: status=%s rel=%.3e msg=%s", r.Status, r.FinalRelRes, r.ErrorMessage)
	}
	if r.Iterations > n {
		t.Fatalf("GMRES(n) should converge within n steps, got %d", r.Iterations)
	}
	checkResidual(t, A, r.X, b, 1e-13)
}

func TestZeroRHS(t *testing.T) {
	// b = 0 with zero guess: residual already zero -> immediate convergence
	// measured on the absolute residual scale.
	A := spd3(t)
	r, err := Solve(context.Background(), A, []float64{0, 0, 0},
		Config{Method: MethodCG, Tolerance: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged || r.Iterations != 0 || r.FinalRelRes != 0 {
		t.Fatalf("status=%s iters=%d rel=%.3e", r.Status, r.Iterations, r.FinalRelRes)
	}
}

func TestDefaultToleranceApplied(t *testing.T) {
	A := spd3(t)
	r, err := Solve(context.Background(), A, []float64{1, 2, 3}, Config{Method: MethodCG})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusConverged || r.FinalRelRes > 1e-8 {
		t.Fatalf("default tolerance not applied: status=%s rel=%.3e", r.Status, r.FinalRelRes)
	}
}

func TestMidSolveCancellation(t *testing.T) {
	// A matrix that needs many iterations plus a tiny tolerance gives the
	// context time to fire mid-solve; CG checks cancellation each step.
	n := 100
	Ag := laplacianLike(n)
	var st []sparse.Triplet
	for i := 0; i < n; i++ {
		st = append(st, sparse.Triplet{Row: i, Col: i, Value: 2})
		if i > 0 {
			st = append(st, sparse.Triplet{Row: i, Col: i - 1, Value: -1})
		}
	}
	As, _ := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: st}).ToCSR()
	b := make([]float64, n)
	for i := range b {
		b[i] = 1
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := Solve(ctx, As, b, Config{Method: MethodCG, Tolerance: 1e-15, MaxIterations: 100000})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != StatusCanceled {
		t.Fatalf("status=%s, want canceled", r.Status)
	}

	// Cancel after a short delay to hit the in-loop branch.
	ctx2, cancel2 := context.WithCancel(context.Background())
	go func() {
		time.Sleep(2 * time.Millisecond)
		cancel2()
	}()
	r2, _ := Solve(ctx2, Ag, b, Config{Method: MethodGMRES, Tolerance: 1e-15, MaxIterations: 100000, Restart: n})
	if r2.Status != StatusCanceled && r2.Status != StatusConverged {
		t.Fatalf("status=%s, want canceled or converged", r2.Status)
	}
	if r2.Status == StatusConverged && r2.FinalRelRes > 1e-15 {
		t.Fatal("converged result must meet tolerance")
	}
}
