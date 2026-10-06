package runner

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// batchTasks builds count well-conditioned tasks that each converge quickly.
func batchTasks(t *testing.T, count, n int) []Task {
	t.Helper()
	tasks := make([]Task, 0, count)
	for k := 0; k < count; k++ {
		trips := make([]sparse.Triplet, 0, 2*n)
		for i := 0; i < n; i++ {
			trips = append(trips, sparse.Triplet{Row: i, Col: i, Value: 4})
			if i+1 < n {
				trips = append(trips, sparse.Triplet{Row: i, Col: i + 1, Value: 1})
			}
		}
		A, err := (&sparse.COO{Rows: n, Cols: n, Symmetric: true, Triplets: trips}).ToCSR()
		if err != nil {
			t.Fatal(err)
		}
		b := make([]float64, n)
		for i := range b {
			b[i] = float64((i+k)%5) + 1
		}
		tasks = append(tasks, Task{
			Name: fmt.Sprintf("task-%d", k),
			A:    A, B: b,
			Cfg: solver.Config{Tolerance: 1e-10, Method: solver.MethodCG, MaxIterations: 500},
		})
	}
	return tasks
}

func statuses(results []BatchResult) []solver.Status {
	out := make([]solver.Status, len(results))
	for i, r := range results {
		out[i] = r.Result.Status
	}
	return out
}

// TestBatchBudgetExhaustion verifies that when the iteration budget runs
// out, unstarted tasks are explicitly marked skipped (never dropped, never
// mislabeled as solver failures) and the order of results is preserved.
func TestBatchBudgetExhaustion(t *testing.T) {
	tasks := batchTasks(t, 6, 10)
	opts := BatchOptions{Workers: 2, SegmentSize: 1,
		Budget: Budget{MaxTotalIterations: 1}}
	results, err := RunBatch(context.Background(), opts, tasks, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != len(tasks) {
		t.Fatalf("got %d results for %d tasks", len(results), len(tasks))
	}
	var paused, skipped int
	for i, r := range results {
		if r.Name != tasks[i].Name {
			t.Fatalf("result %d out of order: %q != %q", i, r.Name, tasks[i].Name)
		}
		if r.Result == nil {
			t.Fatalf("task %s has no result", r.Name)
		}
		switch r.Result.Status {
		case solver.StatusPaused:
			paused++
			if r.Checkpoint == nil {
				t.Fatalf("task %s paused without checkpoint", r.Name)
			}
		case solver.StatusSkipped:
			skipped++
			if r.Checkpoint != nil {
				t.Fatalf("task %s skipped but carries a checkpoint", r.Name)
			}
			if r.Err != nil {
				t.Fatalf("task %s skipped but reports a solver error: %v", r.Name, r.Err)
			}
		default:
			t.Fatalf("task %s: unexpected status %s", r.Name, r.Result.Status)
		}
		// Skipped/paused must never look successful.
		if r.Record().Converged {
			t.Fatalf("task %s: %s record claims convergence", r.Name, r.Result.Status)
		}
	}
	if paused == 0 {
		t.Fatal("expected at least one paused task")
	}
	if skipped == 0 {
		t.Fatal("expected at least one skipped task")
	}
	if got := statuses(results); len(got) != len(tasks) {
		t.Fatal("task silently disappeared")
	}
}

// TestBatchResumeNoRecompute verifies that resuming a batch never recomputes
// finished tasks and continues the unfinished ones to the same conclusion a
// fresh one-shot solve would reach.
func TestBatchResumeNoRecompute(t *testing.T) {
	tasks := batchTasks(t, 5, 10)

	// Reference: one-shot conclusions per task.
	ref := make([]*solver.Result, len(tasks))
	for i, tk := range tasks {
		r, err := solver.Solve(context.Background(), tk.A, tk.B, tk.Cfg)
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != solver.StatusConverged {
			t.Fatalf("reference solve did not converge: %s", r.Status)
		}
		ref[i] = r
	}

	// Run 1: budget for a single iteration — everything pauses or skips.
	run1, err := RunBatch(context.Background(), BatchOptions{
		Workers: 1, SegmentSize: 1, Budget: Budget{MaxTotalIterations: 1},
	}, tasks, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Run 2: unlimited budget, resuming from run 1.
	run2, err := RunBatch(context.Background(), BatchOptions{
		Workers: 3, SegmentSize: 2,
	}, tasks, run1)
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range run2 {
		if r.Result.Status != solver.StatusConverged {
			t.Fatalf("task %s: status %s after resume", r.Name, r.Result.Status)
		}
		if r.Result.Iterations != ref[i].Iterations {
			t.Fatalf("task %s: resumed iterations %d != one-shot %d",
				r.Name, r.Result.Iterations, ref[i].Iterations)
		}
		if diff := r.Result.FinalRelRes - ref[i].FinalRelRes; diff < -1e-14 || diff > 1e-14 {
			t.Fatalf("task %s: resumed residual %g != one-shot %g",
				r.Name, r.Result.FinalRelRes, ref[i].FinalRelRes)
		}
	}

	// Run 3: everything already terminal — results must be carried over
	// untouched (identical result objects, zero new work).
	run3, err := RunBatch(context.Background(), BatchOptions{
		Workers: 2, Budget: Budget{MaxTotalIterations: 1},
	}, tasks, run2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range run3 {
		if run3[i].Result != run2[i].Result {
			t.Fatalf("task %s was recomputed instead of carried over", tasks[i].Name)
		}
	}
}

// TestBatchTaskIsolation verifies that an invalid task and a canceled task
// do not affect the conclusions of the others.
func TestBatchTaskIsolation(t *testing.T) {
	tasks := batchTasks(t, 4, 10)
	tasks[1].B = tasks[1].B[:3] // invalid: wrong length
	tasks[1].Name = "broken"
	results, err := RunBatch(context.Background(), BatchOptions{Workers: 2}, tasks, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range results {
		if i == 1 {
			if r.Result.Status != solver.StatusInvalidInput || r.Err == nil {
				t.Fatalf("broken task: status %s err %v", r.Result.Status, r.Err)
			}
			continue
		}
		if r.Result.Status != solver.StatusConverged {
			t.Fatalf("task %s poisoned by sibling: %s", r.Name, r.Result.Status)
		}
	}
}

// TestBatchTimeBudget verifies that expiring the wall-clock budget stops the
// batch cleanly: unfinished tasks are resumable (canceled with checkpoint or
// skipped), and a later run finishes them.
func TestBatchTimeBudget(t *testing.T) {
	tasks := batchTasks(t, 8, 60)
	run1, err := RunBatch(context.Background(), BatchOptions{
		Workers: 2, SegmentSize: 3,
		Budget: Budget{MaxElapsed: time.Nanosecond},
	}, tasks, nil)
	if err != nil {
		t.Fatal(err)
	}
	unfinished := 0
	for i, r := range run1 {
		if r.Name != tasks[i].Name {
			t.Fatalf("result %d out of order", i)
		}
		switch r.Result.Status {
		case solver.StatusConverged:
		case solver.StatusCanceled, solver.StatusPaused:
			unfinished++
			if r.Checkpoint == nil {
				t.Fatalf("task %s %s without checkpoint", r.Name, r.Result.Status)
			}
		case solver.StatusSkipped:
			unfinished++
		default:
			t.Fatalf("task %s: unexpected status %s", r.Name, r.Result.Status)
		}
	}
	if unfinished == 0 {
		t.Fatal("expired budget should leave tasks unfinished")
	}
	run2, err := RunBatch(context.Background(), BatchOptions{Workers: 2, SegmentSize: 4}, tasks, run1)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range run2 {
		if r.Result.Status != solver.StatusConverged {
			t.Fatalf("task %s: status %s after resume", r.Name, r.Result.Status)
		}
	}
}

// TestBatchConfigValidation rejects nonsensical options.
func TestBatchConfigValidation(t *testing.T) {
	tasks := batchTasks(t, 2, 4)
	if _, err := RunBatch(context.Background(), BatchOptions{Workers: 0}, tasks, nil); err == nil {
		t.Fatal("expected error for non-positive workers")
	}
	if _, err := RunBatch(context.Background(), BatchOptions{Workers: 1}, tasks, []BatchResult{tasksResult(tasks[0])}); err == nil {
		t.Fatal("expected error for mismatched prev length")
	}
}

func tasksResult(t Task) BatchResult {
	return BatchResult{Name: t.Name, Result: &solver.Result{Status: solver.StatusConverged}}
}
