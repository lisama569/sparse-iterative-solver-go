package runner

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// describeTask logs the evidence required when a solve task is judged:
// input size, a convergence-history summary and the decision basis.
func describeTask(t *testing.T, tr TaskResult, n int) {
	t.Helper()
	r := tr.Result
	histSummary := "<none>"
	if r != nil && len(r.History) > 0 {
		histSummary = fmt.Sprintf("first=%.3e last=%.3e len=%d",
			r.History[0], r.History[len(r.History)-1], len(r.History))
	}
	status := "<nil result>"
	rel := "NaN"
	iters := 0
	msg := ""
	if r != nil {
		status = r.Status.String()
		rel = fmt.Sprintf("%.3e", r.FinalRelRes)
		iters = r.Iterations
		msg = r.ErrorMessage
	}
	t.Logf("task=%q size=%d status=%s iters=%d relRes=%s history={%s} basis=%q err=%v",
		tr.Name, n, status, iters, rel, histSummary, msg, tr.Err)
}

// laplacian builds an n x n SPD tridiagonal matrix (-1,2,-1).
func laplacian(t *testing.T, n int) *sparse.CSR {
	t.Helper()
	trips := make([]sparse.Triplet, 0, 3*n)
	for i := 0; i < n; i++ {
		trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 2})
		if i > 0 {
			trips = append(trips, sparse.Triplet{Row: i, Col: i - 1, Value: -1})
		}
	}
	m, err := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: trips}).ToCSR()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func tasksForSizes(t *testing.T, sizes []int) []Task {
	tasks := make([]Task, 0, len(sizes))
	for _, n := range sizes {
		A := laplacian(t, n)
		b := make([]float64, n)
		for i := range b {
			b[i] = 1
		}
		tasks = append(tasks, Task{
			Name: fmt.Sprintf("laplacian-%d", n),
			A:    A, B: b,
			Cfg: solver.Config{Tolerance: 1e-9, Method: solver.MethodCG},
		})
	}
	return tasks
}

func TestRunAllVariousSizesAndWorkers(t *testing.T) {
	sizes := []int{2, 5, 16, 64, 128}
	for _, workers := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			tasks := tasksForSizes(t, sizes)
			results, err := RunAll(context.Background(), workers, tasks)
			if err != nil {
				t.Fatal(err)
			}
			if len(results) != len(tasks) {
				t.Fatalf("got %d results, want %d", len(results), len(tasks))
			}
			for i, tr := range results {
				describeTask(t, tr, sizes[i])
				if tr.Err != nil {
					t.Fatalf("task %s unexpected err %v", tr.Name, tr.Err)
				}
				if tr.Result.Status != solver.StatusConverged {
					t.Fatalf("task %s status=%s rel=%.3e", tr.Name, tr.Result.Status, tr.Result.FinalRelRes)
				}
				if tr.Result.FinalRelRes > tasks[i].Cfg.Tolerance {
					t.Fatalf("task %s rel %.3e above tol", tr.Name, tr.Result.FinalRelRes)
				}
				// Residual evidence.
				Ax, _ := tasks[i].A.MatVec(tr.Result.X)
				var rr, bn float64
				for k := range tasks[i].B {
					rr += (Ax[k] - tasks[i].B[k]) * (Ax[k] - tasks[i].B[k])
					bn += tasks[i].B[k] * tasks[i].B[k]
				}
				rel := math.Sqrt(rr / bn)
				if math.IsNaN(rel) || rel > tasks[i].Cfg.Tolerance {
					t.Fatalf("task %s measured rel %.3e", tr.Name, rel)
				}
			}
		})
	}
}

func TestDeterministicAcrossRuns(t *testing.T) {
	tasks := tasksForSizes(t, []int{32, 64})
	r1, _ := RunAll(context.Background(), 3, tasks)
	r2, _ := RunAll(context.Background(), 3, tasks)
	for i := range r1 {
		if r1[i].Result.Iterations != r2[i].Result.Iterations ||
			r1[i].Result.FinalRelRes != r2[i].Result.FinalRelRes ||
			r1[i].Result.Status != r2[i].Result.Status {
			t.Fatalf("task %d not deterministic: %+v vs %+v", i, r1[i].Result, r2[i].Result)
		}
	}
}

func TestFailureIsolation(t *testing.T) {
	good := tasksForSizes(t, []int{32})[0]
	bad := Task{
		Name: "bad-shape",
		A:    laplacian(t, 4),
		B:    []float64{1, 2}, // length mismatch
		Cfg:  solver.Config{Tolerance: 1e-9},
	}
	maxed := tasksForSizes(t, []int{64})[0]
	maxed.Name = "one-step"
	maxed.Cfg.MaxIterations = 1
	maxed.Cfg.Tolerance = 1e-15

	results, err := RunAll(context.Background(), 2, []Task{good, bad, maxed})
	if err != nil {
		t.Fatal(err)
	}
	describeTask(t, results[0], 32)
	describeTask(t, results[1], 4)
	describeTask(t, results[2], 64)

	if results[0].Result.Status != solver.StatusConverged {
		t.Fatalf("good task contaminated: %s", results[0].Result.Status)
	}
	if results[1].Result.Status != solver.StatusInvalidInput || results[1].Err == nil {
		t.Fatalf("bad task: status=%s err=%v", results[1].Result.Status, results[1].Err)
	}
	var ie *sparse.InputError
	if !errors.As(results[1].Err, &ie) || ie.Kind != sparse.KindStructure {
		t.Fatalf("bad task error not classified as structure: %v", results[1].Err)
	}
	if results[2].Result.Status != solver.StatusMaxIterations {
		t.Fatalf("maxed task: status=%s, want max_iterations", results[2].Result.Status)
	}
	if results[2].Result.Status == solver.StatusConverged {
		t.Fatal("a residual above tolerance must never be success")
	}
}

func TestCancellationDoesNotPoisonOthers(t *testing.T) {
	pool, err := NewPool(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Shutdown()

	// Canceled context submitted before the work runs.
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	big := tasksForSizes(t, []int{256})[0]
	big.Name = "canceled"
	fCanceled, err := pool.Submit(canceledCtx, big)
	if err != nil {
		t.Fatal(err)
	}

	good := tasksForSizes(t, []int{16})[0]
	good.Name = "survivor"
	fGood, err := pool.Submit(context.Background(), good)
	if err != nil {
		t.Fatal(err)
	}

	trC, _ := fCanceled.Get(context.Background())
	trG, _ := fGood.Get(context.Background())
	describeTask(t, *trC, 256)
	describeTask(t, *trG, 16)
	if trC.Result.Status != solver.StatusCanceled {
		t.Fatalf("canceled task status=%s", trC.Result.Status)
	}
	if trG.Result.Status != solver.StatusConverged {
		t.Fatalf("survivor task status=%s msg=%s", trG.Result.Status, trG.Result.ErrorMessage)
	}
}

func TestGetContextCancel(t *testing.T) {
	pool, _ := NewPool(1, 2)
	defer pool.Shutdown()

	// Occupy the worker deterministically with a held entry, then submit
	// the task whose result we wait for (it stays queued).
	release := make(chan struct{})
	pool.jobs <- &entry{
		ctx:  context.Background(),
		done: make(chan *TaskResult, 1),
		task: Task{Name: "holder"},
		hold: release,
	}
	time.Sleep(20 * time.Millisecond) // let the worker pick up the holder
	queued := tasksForSizes(t, []int{32})[0]
	f, err := pool.Submit(context.Background(), queued)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := f.Get(ctx); err == nil {
		t.Fatal("expected Get to return context deadline while task is queued")
	}
	// Release the worker; the previously queued task must still run and
	// its result remain collectible.
	close(release)
	tr, err := f.Get(context.Background())
	if err != nil || tr.Result.Status != solver.StatusConverged {
		t.Fatalf("task lost after Get cancellation: %v %s", err, statusOf(tr))
	}
}

func TestBoundedQueueBackpressure(t *testing.T) {
	const workers, capQ = 1, 1
	pool, err := NewPool(workers, capQ)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Shutdown()

	// Occupy the single worker (via hold) and fill the single queue slot,
	// so the test stays deterministic without depending on solve timing.
	release := make(chan struct{})
	pool.jobs <- &entry{
		ctx:  context.Background(),
		done: make(chan *TaskResult, 1),
		task: Task{Name: "holding-worker"},
		hold: release,
	}
	// Give the worker a moment to pick up the holding entry before we use
	// the one remaining slot for the queue entry.
	time.Sleep(20 * time.Millisecond)
	pool.jobs <- &entry{
		ctx:  context.Background(),
		done: make(chan *TaskResult, 1),
		task: Task{Name: "filling-queue"},
	}

	// A third Submit must block; a canceled context escapes the wait.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err = pool.Submit(ctx, tasksForSizes(t, []int{8})[0])
	elapsed := time.Since(start)
	if err == nil || !strings.Contains(err.Error(), "deadline") {
		t.Fatalf("expected deadline from full queue, got %v", err)
	}
	if elapsed < 40*time.Millisecond {
		t.Fatalf("Submit did not apply backpressure, returned after %v", elapsed)
	}
	close(release)
}

func TestSubmitAfterShutdown(t *testing.T) {
	pool, _ := NewPool(1, 1)
	pool.Shutdown()
	if _, err := pool.Submit(context.Background(), tasksForSizes(t, []int{4})[0]); !errors.Is(err, ErrPoolStopped) {
		t.Fatalf("got %v, want ErrPoolStopped", err)
	}
}

func TestBadPoolConfig(t *testing.T) {
	if _, err := NewPool(0, 1); !errors.Is(err, ErrPoolConfig) {
		t.Fatalf("workers: %v", err)
	}
	if _, err := NewPool(1, 0); !errors.Is(err, ErrPoolConfig) {
		t.Fatalf("queue: %v", err)
	}
}

func statusOf(tr *TaskResult) string {
	if tr == nil || tr.Result == nil {
		return "<nil>"
	}
	return tr.Result.Status.String()
}
