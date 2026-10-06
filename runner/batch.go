package runner

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
)

// Budget caps the total work or wall time of a batch run. A zero field means
// "no limit" for that dimension.
type Budget struct {
	// MaxTotalIterations caps the total solver iterations charged across all
	// tasks of one RunBatch call. Tasks not started when the budget runs out
	// are marked StatusSkipped; a task already in progress is paused with a
	// resumable checkpoint (when SegmentSize bounds the segments).
	MaxTotalIterations int64
	// MaxElapsed caps the wall time of one RunBatch call. When it expires,
	// in-flight tasks are canceled (their checkpoints preserve progress) and
	// tasks not yet started are marked StatusSkipped.
	MaxElapsed time.Duration
}

// BatchOptions configures RunBatch.
type BatchOptions struct {
	// Workers is the number of concurrent solvers; must be positive.
	Workers int
	// Budget is the shared batch budget.
	Budget Budget
	// SegmentSize bounds the iterations a task runs between batch-budget
	// checks. <= 0 runs each task as a single segment (the iteration budget
	// is then only checked between tasks, never mid-task).
	SegmentSize int
}

// BatchResult is the outcome of one task in a batch.
type BatchResult struct {
	Name   string
	Result *solver.Result
	// Checkpoint, when non-nil, resumes a paused or canceled task in a later
	// RunBatch call. Terminal results carry a nil checkpoint.
	Checkpoint *solver.Checkpoint
	Err      error
}

// Record returns the archivable verdict for this task.
func (br *BatchResult) Record() solver.Record {
	return br.Result.Record()
}

// terminal reports whether a solver status is final (no resume can change
// it). Paused, canceled and skipped tasks may continue in a later run.
func terminal(s solver.Status) bool {
	switch s {
	case solver.StatusConverged, solver.StatusMaxIterations,
		solver.StatusBreakdown, solver.StatusInvalidInput:
		return true
	default:
		return false
	}
}

// RunBatch runs tasks with at most opts.Workers concurrent solvers under a
// shared budget and returns one result per task, in submission order.
//
// prev, when non-nil, must be the result slice of a previous RunBatch call
// over the same task list (aligned by index). Tasks that already reached a
// terminal status are carried over untouched and are never recomputed;
// paused or canceled tasks resume from their checkpoint; skipped tasks start
// fresh. A task that exceeds the budget, is skipped or is canceled never
// affects the outcome of any other task.
func RunBatch(ctx context.Context, opts BatchOptions, tasks []Task, prev []BatchResult) ([]BatchResult, error) {
	if opts.Workers <= 0 {
		return nil, fmt.Errorf("%w: workers must be positive, got %d", ErrPoolConfig, opts.Workers)
	}
	if prev != nil && len(prev) != len(tasks) {
		return nil, fmt.Errorf("%w: previous results length %d does not match %d tasks",
			ErrPoolConfig, len(prev), len(tasks))
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if opts.Budget.MaxElapsed > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Budget.MaxElapsed)
		defer cancel()
	}

	results := make([]BatchResult, len(tasks))
	start := make([]*solver.Checkpoint, len(tasks))
	for i := range tasks {
		if prev == nil {
			continue
		}
		pr := prev[i]
		if pr.Result != nil && terminal(pr.Result.Status) {
			// Already final: carry over without recomputing.
			pr.Name = tasks[i].Name
			results[i] = pr
			continue
		}
		if pr.Checkpoint != nil {
			start[i] = pr.Checkpoint
		}
	}

	var consumed atomic.Int64
	limit := opts.Budget.MaxTotalIterations
	budgetLeft := func() (int64, bool) {
		if limit <= 0 {
			return 1 << 62, true
		}
		left := limit - consumed.Load()
		return left, left > 0
	}

	var next atomic.Int64
	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for {
			i := int(next.Add(1) - 1)
			if i >= len(tasks) {
				return
			}
			if results[i].Result != nil {
				continue // carried over from prev
			}
			results[i] = runBatchTask(ctx, opts, tasks[i], start[i], &consumed, budgetLeft)
		}
	}
	for w := 0; w < opts.Workers; w++ {
		wg.Add(1)
		go worker()
	}
	wg.Wait()
	return results, nil
}

// runBatchTask advances one task in budget-checked segments until it reaches
// a terminal status, the batch budget runs out, or the context is canceled.
func runBatchTask(ctx context.Context, opts BatchOptions, t Task, ckpt *solver.Checkpoint,
	consumed *atomic.Int64, budgetLeft func() (int64, bool)) BatchResult {

	resume := ckpt != nil
	for {
		left, ok := budgetLeft()
		if !ok {
			if !resume {
				// Never started: mark explicitly instead of dropping it.
				return BatchResult{Name: t.Name, Result: &solver.Result{
					Status:       solver.StatusSkipped,
					ErrorMessage: "batch iteration budget exhausted before task started",
				}}
			}
			// Started but the batch budget is spent: pause with progress.
			return BatchResult{Name: t.Name, Checkpoint: ckpt, Result: &solver.Result{
				Status:       solver.StatusPaused,
				Tolerance:    ckpt.Tolerance,
				Iterations:   ckpt.Iterations,
				FinalRelRes:  ckpt.Rel,
				History:      append([]float64(nil), ckpt.History...),
				ErrorMessage: "batch iteration budget exhausted mid-task",
			}}
		}
		seg := int64(opts.SegmentSize)
		if seg <= 0 || seg > left {
			seg = left
		}
		before := 0
		if ckpt != nil {
			before = ckpt.Iterations
		}
		res, next, err := solver.SolveSegment(ctx, t.A, t.B, t.Cfg, ckpt, int(seg))
		if res != nil {
			consumed.Add(int64(res.Iterations - before))
		}
		if err != nil {
			// Invalid input or a checkpoint handoff error: terminal for this
			// task, and the previous checkpoint (if any) is preserved.
			return BatchResult{Name: t.Name, Result: res, Checkpoint: next, Err: err}
		}
		switch res.Status {
		case solver.StatusPaused:
			ckpt = next
			resume = true
			// Loop: re-check the batch budget, then continue.
		case solver.StatusCanceled:
			if next == nil && res.Iterations == 0 {
				// Canceled before making any progress: the task never
				// really started, so mark it skipped rather than canceled.
				return BatchResult{Name: t.Name, Result: &solver.Result{
					Status:       solver.StatusSkipped,
					Tolerance:    res.Tolerance,
					ErrorMessage: "batch context canceled before the task started",
				}}
			}
			return BatchResult{Name: t.Name, Result: res, Checkpoint: next}
		default:
			return BatchResult{Name: t.Name, Result: res}
		}
	}
}
