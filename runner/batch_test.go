package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

func TestRunBatchOrderAndConvergence(t *testing.T) {
	sizes := []int{4, 8, 16, 32, 64}
	tasks := tasksForSizes(t, sizes)
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(sum.Results) != len(tasks) {
		t.Fatalf("results %d tasks %d", len(sum.Results), len(tasks))
	}
	for i := range sum.Results {
		if sum.Results[i].Name != tasks[i].Name {
			t.Fatalf("order broken at %d: %q", i, sum.Results[i].Name)
		}
		if sum.Results[i].Result.Status != solver.StatusConverged {
			t.Fatalf("task %d status=%s", i, sum.Results[i].Result.Status)
		}
	}
	if sum.Converged != len(tasks) || sum.Skipped != 0 || sum.Failed != 0 {
		t.Fatalf("counts=%+v", sum)
	}
}

// TestRunBatchIterationBudgetSkips: shared iteration cap; tasks that do not
// get a turn are explicitly skipped, earlier tasks are unaffected.
func TestRunBatchIterationBudgetSkips(t *testing.T) {
	sizes := []int{128, 128, 16, 16}
	tasks := tasksForSizes(t, sizes)
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	// Tiny budget: only the first task(s) can start.
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, SegmentSteps: 5, MaxTotalIterations: 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	// With workers=1 and ordered dispatch, task 0 consumes the whole 5-step
	// budget; tasks 1..3 never start and must be skipped (not failed).
	if sum.Results[0].Result.Status != solver.StatusPaused {
		t.Fatalf("task0 status=%s want paused", sum.Results[0].Result.Status)
	}
	if sum.Results[0].Result.Progress == nil {
		t.Fatal("paused budget-exhausted task must carry progress")
	}
	if sum.Results[0].Result.Iterations != 5 {
		t.Fatalf("task0 iters=%d want 5", sum.Results[0].Result.Iterations)
	}
	for i := 1; i < len(tasks); i++ {
		tr := sum.Results[i]
		if tr.State != StateSkipped || tr.Result.Status != solver.StatusSkipped {
			t.Fatalf("task %d state=%s status=%s, want skipped", i, tr.State, tr.Result.Status)
		}
		if !errors.Is(skipErr(tr.Result.ErrorMessage), ErrBatchBudgetExhausted) {
			t.Fatalf("task %d skip reason=%q", i, tr.Result.ErrorMessage)
		}
		if tr.Result.FinalRelRes != 0 || tr.Result.Iterations != 0 {
			t.Fatal("skipped task must not pretend to have solved anything")
		}
	}
	if sum.Skipped != 3 || sum.Paused != 1 {
		t.Fatalf("counts=%+v", sum)
	}
	if sum.IterationsUsed != 5 {
		t.Fatalf("used=%d want 5", sum.IterationsUsed)
	}
}

func skipErr(msg string) error {
	switch {
	case strings.Contains(msg, "budget"):
		return ErrBatchBudgetExhausted
	case strings.Contains(msg, "time"):
		return ErrBatchDeadline
	default:
		return ErrBatchCanceled
	}
}

// TestRunBatchResumeDoesNotRecompute: second run with Previous reuses
// terminal results and continues only unfinished ones.
func TestRunBatchResumeDoesNotRecompute(t *testing.T) {
	tasks := tasksForSizes(t, []int{64, 64, 32, 32})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	first, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, SegmentSteps: 5, MaxTotalIterations: 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.Results[0].Result.Status != solver.StatusPaused {
		t.Fatalf("task0 should be paused, got %s", first.Results[0].Result.Status)
	}

	// Pretend task 2 finished earlier and task 3 was invalid; mix into
	// Previous to prove reuse is honored regardless of neighbors.
	done, _ := RunBatch(context.Background(), BatchRequest{
		Tasks: []Task{tasks[2]}, Cfg: BatchConfig{Workers: 1},
	})
	prev := make([]TaskResult, len(tasks))
	copy(prev, first.Results)
	prev[2] = done.Results[0]
	prev[3] = TaskResult{Name: "t3", State: StateComputed,
		Result: &solver.Result{Status: solver.StatusInvalidInput,
			ErrorMessage: "synthetic prior failure"}}

	second, err := RunBatch(context.Background(), BatchRequest{
		Tasks:    tasks,
		Cfg:      BatchConfig{Workers: 2},
		Previous: prev,
	})
	if err != nil {
		t.Fatal(err)
	}

	// task 0 must CONTINUE from 5 steps, not restart: total iters matches
	// a fresh one-shot solve and segment of run 2 starts at 5.
	oneShot, _ := RunBatch(context.Background(), BatchRequest{
		Tasks: []Task{tasks[0]}, Cfg: BatchConfig{Workers: 1},
	})
	r0 := second.Results[0]
	if r0.Result.Status != solver.StatusConverged {
		t.Fatalf("task0 resumed status=%s msg=%s", r0.Result.Status, r0.Result.ErrorMessage)
	}
	if r0.Result.Iterations != oneShot.Results[0].Result.Iterations {
		t.Fatalf("task0 resumed iters=%d oneshot=%d", r0.Result.Iterations, oneShot.Results[0].Result.Iterations)
	}
	// task 1 was skipped in run 1; it runs fresh in run 2.
	if second.Results[1].State != StateComputed ||
		second.Results[1].Result.Status != solver.StatusConverged {
		t.Fatalf("task1 state=%s status=%s", second.Results[1].State, second.Results[1].Result.Status)
	}
	// tasks 2 and 3 reused verbatim.
	if second.Results[2].State != StateReused {
		t.Fatalf("task2 state=%s want reused", second.Results[2].State)
	}
	if second.Results[2].Result != done.Results[0].Result {
		t.Fatal("reused result should be the same record, not recomputed")
	}
	if second.Results[3].State != StateReused ||
		second.Results[3].Result.Status != solver.StatusInvalidInput {
		t.Fatalf("task3 state=%s status=%s", second.Results[3].State, second.Results[3].Result.Status)
	}
	// Reused tasks consume no budget in run 2.
	if second.Results[2].Duration != 0 || second.Results[3].Duration != 0 {
		t.Fatal("reused tasks must spend no compute time")
	}
}

// TestRunBatchFailureIsolation: one bad task must not change neighbors.
func TestRunBatchFailureIsolation(t *testing.T) {
	good := tasksForSizes(t, []int{32})[0]
	bad := Task{Name: "bad", A: laplacian(t, 4), B: []float64{1, 2},
		Cfg: solver.Config{Tolerance: 1e-9}}
	maxed := tasksForSizes(t, []int{64})[0]
	maxed.Name = "maxed"
	maxed.Cfg.MaxIterations = 1
	maxed.Cfg.Tolerance = 1e-15
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: []Task{good, bad, maxed}, Cfg: BatchConfig{Workers: 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if sum.Results[0].Result.Status != solver.StatusConverged {
		t.Fatalf("good contaminated: %s", sum.Results[0].Result.Status)
	}
	if sum.Results[1].Result.Status != solver.StatusInvalidInput || sum.Results[1].Err == nil {
		t.Fatalf("bad status=%s", sum.Results[1].Result.Status)
	}
	if sum.Results[2].Result.Status != solver.StatusMaxIterations {
		t.Fatalf("maxed status=%s", sum.Results[2].Result.Status)
	}
	if sum.Converged != 1 || sum.Failed != 2 {
		t.Fatalf("counts=%+v", sum)
	}
}

// TestRunBatchContextCancelMarksUnstarted: cancel before dispatch; unstarted
// tasks are skipped with a clear reason, in-flight progress is retained.
func TestRunBatchContextCancelMarksUnstarted(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	tasks := tasksForSizes(t, []int{256, 16, 16})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	cancel()
	sum, err := RunBatch(ctx, BatchRequest{
		Tasks: tasks, Cfg: BatchConfig{Workers: 1, SegmentSteps: 5},
	})
	if err != nil {
		t.Fatal(err)
	}
	// All tasks are either canceled (in-flight with progress) or skipped;
	// none may be reported as a solver failure or silently dropped.
	seen := map[solver.Status]int{}
	for i, tr := range sum.Results {
		seen[tr.Result.Status]++
		if tr.Result.Status != solver.StatusCanceled && tr.Result.Status != solver.StatusSkipped {
			t.Fatalf("task %d unexpected status %s", i, tr.Result.Status)
		}
	}
	if seen[solver.StatusSkipped] == 0 {
		t.Fatalf("expected at least one skipped task, got %+v", seen)
	}
}

// TestRunBatchDeadline: a positive but tiny time budget lets early tasks
// finish and marks later tasks skipped by deadline.
func TestRunBatchDeadline(t *testing.T) {
	tasks := tasksForSizes(t, []int{8, 8, 8, 8})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
		tasks[i].Cfg.Tolerance = 1e-14
	}
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, MaxDuration: time.Nanosecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	skippedDeadline := 0
	for _, tr := range sum.Results {
		if tr.State == StateSkipped && strings.Contains(tr.Result.ErrorMessage, "time") {
			skippedDeadline++
		}
	}
	if skippedDeadline == 0 {
		t.Fatalf("expected deadline skips, results=%+v", statuses(sum))
	}
}

func statuses(sum *BatchSummary) []string {
	out := make([]string, len(sum.Results))
	for i := range sum.Results {
		out[i] = sum.Results[i].Result.Status.String()
	}
	return out
}

// TestRunBatchRecordsArchivable: Records() yields complete JSON-able reports
// in order including skipped entries.
func TestRunBatchRecordsArchivable(t *testing.T) {
	tasks := tasksForSizes(t, []int{16, 16, 16})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, SegmentSteps: 3, MaxTotalIterations: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	recs := sum.Records()
	if len(recs) != 3 {
		t.Fatalf("records=%d", len(recs))
	}
	raw, err := json.Marshal(recs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "skipped") || !strings.Contains(string(raw), "paused") {
		t.Fatalf("archive missing statuses: %s", raw)
	}
	for i, r := range recs {
		if r.Name != tasks[i].Name {
			t.Fatalf("record %d name=%q", i, r.Name)
		}
		if r.Status == "converged" && !r.ResidualMet {
			t.Fatal("converged record inconsistent")
		}
		if r.Status != "converged" && r.Success {
			t.Fatalf("%s record marked success", r.Status)
		}
	}
	// Round-trip.
	var back []solver.Report
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if len(back) != 3 {
		t.Fatalf("round trip len=%d", len(back))
	}
}

// TestRunBatchBudgetRespectsMeasuredResidual: converged tasks must actually
// meet the tolerance; budget accounting sums the segment steps.
func TestRunBatchBudgetRespectsMeasuredResidual(t *testing.T) {
	tasks := tasksForSizes(t, []int{12, 24})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("t%d", i)
	}
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks, Cfg: BatchConfig{Workers: 2, SegmentSteps: 7},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, tr := range sum.Results {
		if tr.Result.Status != solver.StatusConverged {
			t.Fatalf("task %d %s", i, tr.Result.Status)
		}
		Ax, _ := tasks[i].A.MatVec(tr.Result.X)
		var rr, bn float64
		for k := range tasks[i].B {
			rr += (Ax[k] - tasks[i].B[k]) * (Ax[k] - tasks[i].B[k])
			bn += tasks[i].B[k] * tasks[i].B[k]
		}
		rel := math.Sqrt(rr / bn)
		if rel > tasks[i].Cfg.Tolerance {
			t.Fatalf("task %d measured rel %.3e > tol", i, rel)
		}
	}
}

// TestRunBatchSkippedRunsFreshNextTime: a task skipped in run 1 starts from
// scratch in run 2 and converges; a paused task continues with cumulative
// iterations equal to a one-shot solve.
func TestRunBatchSkippedRunsFreshNextTime(t *testing.T) {
	// Non-symmetric general matrix to exercise GMRES segmentation in batch.
	genTasks := func() []Task {
		sizes := []int{30, 30, 30, 30}
		out := make([]Task, len(sizes))
		for i, n := range sizes {
			A := generalLaplacian(n)
			b := make([]float64, n)
			for k := range b {
				b[k] = float64(k%4 - 1)
			}
			out[i] = Task{Name: fmt.Sprintf("g%d", i), A: A, B: b,
				Cfg: solver.Config{Tolerance: 1e-8, Method: solver.MethodGMRES,
					MaxIterations: 2000, Restart: 12}}
		}
		return out
	}

	tasks := genTasks()
	first, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, SegmentSteps: 4, MaxTotalIterations: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.Results[0].Result.Status != solver.StatusPaused {
		t.Fatalf("task0=%s", first.Results[0].Result.Status)
	}
	for i := 1; i < len(tasks); i++ {
		if first.Results[i].State != StateSkipped {
			t.Fatalf("task %d state=%s want skipped", i, first.Results[i].State)
		}
	}

	// Reference one-shot batch, no budget.
	ref, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks, Cfg: BatchConfig{Workers: 2},
	})
	if err != nil {
		t.Fatal(err)
	}

	second, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks, Cfg: BatchConfig{Workers: 2, SegmentSteps: 4},
		Previous: first.Results,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range tasks {
		got := second.Results[i]
		want := ref.Results[i]
		if got.Result.Status != solver.StatusConverged {
			t.Fatalf("task %d status=%s msg=%s", i, got.Result.Status, got.Result.ErrorMessage)
		}
		if got.Result.Iterations != want.Result.Iterations {
			t.Fatalf("task %d iters resumed=%d oneshot=%d",
				i, got.Result.Iterations, want.Result.Iterations)
		}
		if math.Abs(got.Result.FinalRelRes-want.Result.FinalRelRes) > 1e-12 {
			t.Fatalf("task %d residual %.6e vs %.6e",
				i, got.Result.FinalRelRes, want.Result.FinalRelRes)
		}
		if i >= 1 && got.State != StateComputed {
			t.Fatalf("previously skipped task %d state=%s want computed", i, got.State)
		}
	}
}

// TestRunBatchPreviousLengthMismatch is rejected without touching tasks.
func TestRunBatchPreviousLengthMismatch(t *testing.T) {
	tasks := tasksForSizes(t, []int{8, 8})
	_, err := RunBatch(context.Background(), BatchRequest{
		Tasks:    tasks,
		Cfg:      BatchConfig{Workers: 1},
		Previous: make([]TaskResult, 1),
	})
	if err == nil || !strings.Contains(err.Error(), "Previous length") {
		t.Fatalf("err=%v", err)
	}
}

// generalLaplacian builds the n x n tridiagonal matrix WITHOUT the symmetric
// flag so MethodGMRES is forced.
func generalLaplacian(n int) *sparse.CSR {
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

// TestRunBatchDeadlineRetainsProgress: with a tiny slice size, the first
// task starts, the deadline fires mid-flight, and its result is a resumable
// canceled-with-progress record (never a failure, never a fake success).
func TestRunBatchDeadlineRetainsProgress(t *testing.T) {
	tasks := tasksForSizes(t, []int{120, 120})
	for i := range tasks {
		tasks[i].Name = fmt.Sprintf("d%d", i)
		tasks[i].Cfg.Tolerance = 1e-15
	}
	sum, err := RunBatch(context.Background(), BatchRequest{
		Tasks: tasks,
		Cfg:   BatchConfig{Workers: 1, SegmentSteps: 1, MaxDuration: time.Millisecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	r0 := sum.Results[0]
	// It either paused past the deadline slice boundary or was canceled with
	// progress; in all non-terminal-but-started cases progress must exist.
	switch r0.Result.Status {
	case solver.StatusPaused, solver.StatusCanceled:
		if r0.Result.Progress == nil {
			t.Fatalf("%s without progress", r0.Result.Status)
		}
		if r0.Result.Iterations == 0 {
			t.Fatalf("%s without any recorded iterations", r0.Result.Status)
		}
	default:
		t.Skipf("task finished before deadline: %s", r0.Result.Status)
	}
	// The rest of the batch must not have silently disappeared.
	for i := 1; i < len(tasks); i++ {
		st := sum.Results[i].Result.Status
		if st != solver.StatusSkipped && st != solver.StatusCanceled {
			t.Fatalf("task %d unexpected %s", i, st)
		}
	}
}
