package solver_test

import (
	"context"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/lisama569/sparse-iterative-solver-go/mmio"
	"github.com/lisama569/sparse-iterative-solver-go/runner"
	"github.com/lisama569/sparse-iterative-solver-go/solver"
)

// TestEndToEndMatrixMarketToRunner exercises the full local pipeline:
// Matrix Market text -> validation -> CSR -> concurrent solves via Pool.
func TestEndToEndMatrixMarketToRunner(t *testing.T) {
	mtx := `%%MatrixMarket matrix coordinate real symmetric
% 5x5 SPD tridiagonal (full diagonal + lower off-diagonals)
5 5 9
1 1 2
2 1 -1
2 2 2
3 2 -1
3 3 2
4 3 -1
4 4 2
5 4 -1
5 5 2
`
	coo, err := mmio.ReadCOO(strings.NewReader(mtx))
	if err != nil {
		t.Fatal(err)
	}
	A, err := coo.ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	if !A.Symmetric || A.NNZ() != 13 {
		t.Fatalf("unexpected matrix: sym=%t nnz=%d", A.Symmetric, A.NNZ())
	}

	// Many tasks share the same immutable A with different RHS and
	// configurations; run with deliberately few workers.
	const numTasks = 64
	tasks := make([]runner.Task, numTasks)
	for i := range tasks {
		b := make([]float64, 5)
		for k := range b {
			b[k] = float64((i + k) % 7)
		}
		cfg := solver.Config{Tolerance: 1e-10}
		if i%2 == 0 {
			cfg.Precondition = true
		}
		tasks[i] = runner.Task{Name: "task", A: A, B: b, Cfg: cfg}
	}
	results, err := runner.RunAll(context.Background(), 3, tasks)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != numTasks {
		t.Fatalf("got %d results", len(results))
	}
	for i, tr := range results {
		if tr.Err != nil {
			t.Fatalf("task %d err: %v", i, tr.Err)
		}
		if tr.Result.Status != solver.StatusConverged {
			t.Fatalf("task %d status=%s rel=%.3e history=%v",
				i, tr.Result.Status, tr.Result.FinalRelRes, tr.Result.History)
		}
		Ax, _ := A.MatVec(tr.Result.X)
		var rr, bn float64
		for k := range tasks[i].B {
			rr += (Ax[k] - tasks[i].B[k]) * (Ax[k] - tasks[i].B[k])
			bn += tasks[i].B[k] * tasks[i].B[k]
		}
		if bn == 0 {
			continue
		}
		rel := math.Sqrt(rr / bn)
		if rel > 1e-10 {
			t.Fatalf("task %d measured residual %.3e", i, rel)
		}
	}
}

// TestEndToEndSharedMatrixHighConcurrency stresses immutable input reuse.
func TestEndToEndSharedMatrixHighConcurrency(t *testing.T) {
	mtx := `%%MatrixMarket matrix coordinate real general
3 3 4
1 1 4
2 1 1
2 2 3
3 3 2
`
	coo, err := mmio.ReadCOO(strings.NewReader(mtx))
	if err != nil {
		t.Fatal(err)
	}
	A, _ := coo.ToCSR()
	b := []float64{1, 2, 3}

	const goroutines = 32
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := solver.Solve(context.Background(), A, b,
				solver.Config{Tolerance: 1e-11, Method: solver.MethodGMRES})
			if err != nil {
				errs <- err
				return
			}
			if r.Status != solver.StatusConverged || r.FinalRelRes > 1e-11 {
				errs <- &unexpectedStatus{r.Status, r.FinalRelRes}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Fatal(e)
	}
}

type unexpectedStatus struct {
	status solver.Status
	rel    float64
}

func (e *unexpectedStatus) Error() string {
	return "unexpected solve outcome " + e.status.String()
}
