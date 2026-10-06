package runner

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
)

// TaskState refines what happened to a task during a batch run.
type TaskState int

const (
	// StateComputed: at least one segment ran in this batch.
	StateComputed TaskState = iota + 1
	// StateReused: the task already had a terminal result in Previous and
	// was not recomputed.
	StateReused
	// StateSkipped: the task never started because the batch budget,
	// deadline or batch context gave out before its turn.
	StateSkipped
)

func (s TaskState) String() string {
	switch s {
	case StateComputed:
		return "computed"
	case StateReused:
		return "reused"
	case StateSkipped:
		return "skipped"
	default:
		return "unknown"
	}
}

// defaultSegmentSteps is the per-slice cap for overnight batch tasks: small
// enough that budget/cancellation checks happen frequently, large enough
// that boundary-measurement overhead is negligible.
const defaultSegmentSteps = 64

// BatchConfig configures an overnight batch run.
type BatchConfig struct {
	// Workers is the fixed worker count; <= 0 selects 1.
	Workers int
	// SegmentSteps caps the matrix-vector steps of one work slice between
	// budget/context checks; <= 0 selects defaultSegmentSteps.
	SegmentSteps int
	// MaxTotalIterations caps total matrix-vector steps across all tasks.
	// <= 0 means unlimited.
	MaxTotalIterations int64
	// MaxDuration caps wall-clock time of the whole batch; <= 0 means
	// unlimited.
	MaxDuration time.Duration
}

// BatchRequest is one overnight batch.
type BatchRequest struct {
	Tasks []Task
	Cfg   BatchConfig
	// Previous, when non-nil, must have the same length and order as Tasks.
	// Terminal results are reused without recomputation; unfinished entries
	// (paused/canceled/skipped/missing) are continued or started.
	Previous []TaskResult
}

// BatchSummary is the archivable summary of a batch run.
type BatchSummary struct {
	// Results are in the exact order the tasks were submitted.
	Results []TaskResult
	// IterationsUsed is the total number of matrix-vector steps actually
	// performed across tasks in THIS run.
	IterationsUsed int64
	// Converged / Failed / Skipped / Paused count final task states.
	Converged, Failed, Skipped, Paused int
}

// Skip reasons recorded on the skipped result.
var (
	ErrBatchBudgetExhausted = errors.New("batch iteration budget exhausted before task started")
	ErrBatchDeadline        = errors.New("batch time budget exhausted before task started")
	ErrBatchCanceled        = errors.New("batch context canceled before task started")
)

// batchJob is one work slice handed to a worker. index is the task index;
// progressive is true when the task carries progress from a prior slice.
type batchJob struct {
	index int
}

// RunBatch runs all tasks under a shared budget, preserving input order and
// per-task isolation, and supports continuation from req.Previous.
//
// Semantics:
//   - Tasks are handed to workers strictly in submission order. When the
//     iteration budget or time budget is exhausted (or the batch context is
//     canceled), every task that has not yet STARTED is reported as
//     StatusSkipped with an explicit reason; it never silently disappears
//     and is never counted as a solve failure.
//   - A task that already holds a terminal (converged/failed) result in
//     Previous is REUSED: it is not recomputed and consumes no budget.
//   - Tasks in progress when a limit gives out finish their current slice
//     and return StatusPaused (resumable) or StatusCanceled with the
//     progress earned; one task exceeding its slice never affects others.
//   - Results always come back in submission order.
func RunBatch(ctx context.Context, req BatchRequest) (*BatchSummary, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	workers := req.Cfg.Workers
	if workers <= 0 {
		workers = 1
	}
	segSteps := req.Cfg.SegmentSteps
	if segSteps <= 0 {
		segSteps = defaultSegmentSteps
	}
	n := len(req.Tasks)
	if req.Previous != nil && len(req.Previous) != n {
		return nil, errors.New("RunBatch: Previous length does not match Tasks length")
	}

	// The caller's ctx and MaxDuration are combined into one context for
	// the compute loops; deadline is retained so skip reasons can name the
	// exact limit that gave out.
	batchCtx := ctx
	cancel := context.CancelFunc(func() {})
	if req.Cfg.MaxDuration > 0 {
		batchCtx, cancel = context.WithTimeout(ctx, req.Cfg.MaxDuration)
	}
	defer cancel()
	var deadline time.Time
	if req.Cfg.MaxDuration > 0 {
		deadline = time.Now().Add(req.Cfg.MaxDuration)
	}

	var used atomic.Int64
	budget := req.Cfg.MaxTotalIterations // <= 0 means unlimited

	results := make([]TaskResult, n)
	type work struct {
		index int
	}
	jobs := make(chan work, workers+1)
	var wg sync.WaitGroup

	prog := make([]*solver.Progress, n)

	worker := func() {
		defer wg.Done()
		for j := range jobs {
			i := j.index
			task := req.Tasks[i]
			start := time.Now()

			// Continue from an unfinished previous result (paused or
			// canceled), if one was handed in. Terminal and skipped
			// entries never reach a worker.
			if req.Previous != nil && req.Previous[i].Result != nil {
				if p := req.Previous[i].Result.Progress; p != nil {
					prog[i] = p
				}
			}

			// Run slice after slice while the task stays pausable and
			// budget/time remain.
			var res *solver.Result
			var serr error
			for {
				if err := batchCtx.Err(); err != nil {
					if prog[i] == nil {
						res = skippedResult(skipErrFromContext(deadline, req.Cfg.MaxDuration > 0), task.Cfg)
					} else {
						res, serr = stoppedResult(err, prog[i]), nil
					}
					break
				}
				if req.Cfg.MaxDuration > 0 && time.Now().After(deadline) {
					if prog[i] == nil {
						res = skippedResult(ErrBatchDeadline, task.Cfg)
					} else {
						res = stoppedResult(ErrBatchDeadline, prog[i])
						res.Status = solver.StatusCanceled
					}
					break
				}
				remaining := int64(-1)
				if budget > 0 {
					remaining = budget - used.Load()
					if remaining <= 0 {
						if prog[i] == nil {
							// Never actually started: a skip, not a solve
							// failure and not a cancellation.
							res = skippedResult(ErrBatchBudgetExhausted, task.Cfg)
						}
						// else res already holds the last segment's full
						// paused/canceled record.
						break
					}
				}
				steps := int64(segSteps)
				if budget > 0 && steps > remaining {
					steps = remaining
				}
				r, err := solver.Advance(batchCtx, task.A, task.B, task.Cfg, prog[i], int(steps))
				serr = err
				res = r
				if r != nil {
					used.Add(int64(r.SegmentIterations))
					if r.Progress != nil {
						prog[i] = r.Progress
					}
				}
				if r == nil || r.Status.Terminal() {
					break
				}
				if r.Status == solver.StatusPaused || r.Status == solver.StatusCanceled {
					// Time/budget recheck: continue the same task in the
					// next slice while limits allow.
					if req.Cfg.MaxDuration > 0 && time.Now().After(deadline) {
						if r.Status == solver.StatusPaused {
							r.Status = solver.StatusCanceled
							r.ErrorMessage = ErrBatchDeadline.Error()
							r.Basis = "batch time budget exhausted after " +
								strconv.Itoa(r.Iterations) +
								" iteration(s); progress retained and resumable"
						}
						break
					}
					if budget > 0 && used.Load() >= budget {
						break // leave StatusPaused so it resumes next run
					}
					if batchCtx.Err() != nil {
						break // retain status (canceled/paused)
					}
					continue
				}
			}

			elapsed := time.Since(start)
			if res == nil {
				res = &solver.Result{Status: solver.StatusBreakdown,
					ErrorMessage: "solver returned no result"}
			}
			state := StateComputed
			if res.Status == solver.StatusSkipped {
				state = StateSkipped
				elapsed = 0
			}
			results[i] = TaskResult{Name: task.Name, Result: res, Err: serr,
				State: state, Duration: elapsed}
		}
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go worker()
	}

	// Feeder: preserves submission order and decides skips.
	stopIndex := -1
	var stopReason error
feeder:
	for i := range req.Tasks {
		// Reuse terminal previous results without spending budget.
		if req.Previous != nil && req.Previous[i].Result != nil {
			pr := req.Previous[i].Result
			if pr.Status.Terminal() {
				results[i] = TaskResult{Name: req.Tasks[i].Name, Result: pr,
					Err: req.Previous[i].Err, State: StateReused}
				continue
			}
		}
		// Can this task still start?
		if err := ctx.Err(); err != nil {
			fillSkipped(results, req.Tasks, i, ErrBatchCanceled)
			break feeder
		}
		if req.Cfg.MaxDuration > 0 && time.Now().After(deadline) {
			fillSkipped(results, req.Tasks, i, ErrBatchDeadline)
			break feeder
		}
		if budget > 0 && used.Load() >= budget {
			fillSkipped(results, req.Tasks, i, ErrBatchBudgetExhausted)
			break feeder
		}
		// Hand the task over in order. Abandon the wait if the batch stops;
		// a task that is actually enqueued still runs and is isolated.
		sent := false
		select {
		case jobs <- work{index: i}:
			sent = true
		case <-batchCtx.Done():
		}
		if !sent {
			// The worker may have raced us; only mark slots that remain
			// empty (after all workers finish, done below).
			stopIndex = i
			stopReason = skipReasonFor(deadline, req.Cfg.MaxDuration > 0)
			break feeder
		}
	}
	close(jobs)
	wg.Wait()
	if stopIndex >= 0 {
		fillSkippedEmpty(results, req.Tasks, stopIndex, stopReason)
	}

	sum := &BatchSummary{Results: results, IterationsUsed: used.Load()}
	for _, tr := range results {
		switch statusOfResult(tr) {
		case solver.StatusConverged:
			sum.Converged++
		case solver.StatusSkipped:
			sum.Skipped++
		case solver.StatusPaused:
			sum.Paused++
		default:
			sum.Failed++
		}
	}
	return sum, nil
}

func statusOfResult(tr TaskResult) solver.Status {
	if tr.Result == nil {
		return solver.StatusBreakdown
	}
	return tr.Result.Status
}

// skipBasis renders an explicit decision basis for never-started tasks.
func skipBasis(reason error) string {
	return "task never started: " + reason.Error()
}

// fillSkipped marks every never-started task from index on.
func fillSkipped(results []TaskResult, tasks []Task, from int, reason error) {
	for i := from; i < len(tasks); i++ {
		results[i] = TaskResult{
			Name:   tasks[i].Name,
			State:  StateSkipped,
			Result: skippedResult(reason, tasks[i].Cfg),
		}
	}
}

// skippedResult builds the complete record of a task that never started: it
// has no iterations, no measured residual, no trajectory and an explicit
// reason. It is never reported as a solve failure.
func skippedResult(reason error, cfg solver.Config) *solver.Result {
	return &solver.Result{
		Status:       solver.StatusSkipped,
		History:      []float64{},
		ErrorMessage: reason.Error(),
		Basis:        skipBasis(reason),
	}
}

// stoppedResult records cancellation for a task that had already earned some
// progress; the checkpoint is retained for a later continuation.
func stoppedResult(err error, p *solver.Progress) *solver.Result {
	if p == nil {
		return &solver.Result{Status: solver.StatusCanceled, History: []float64{},
			ErrorMessage: err.Error(),
			Basis:        "batch stopped before the task advanced: " + err.Error()}
	}
	return &solver.Result{
		Status:       solver.StatusCanceled,
		Iterations:   p.Iterations(),
		FinalRelRes:  p.Residual(),
		Progress:     p,
		History:      []float64{},
		ErrorMessage: err.Error(),
		Basis:        "batch stopped with progress retained: " + err.Error(),
	}
}

// skipErrFromContext names the effective stop reason (deadline vs cancel).
func skipErrFromContext(deadline time.Time, hasDeadline bool) error {
	if hasDeadline && time.Now().After(deadline) {
		return ErrBatchDeadline
	}
	return ErrBatchCanceled
}

// IsTerminal reports whether a result already reached a final conclusion.
func IsTerminal(tr TaskResult) bool {
	return tr.Result != nil && tr.Result.Status.Terminal()
}

// fillSkippedEmpty marks every still-empty slot from index from as skipped.
// It runs after workers have drained, so a task that raced into execution
// keeps its real verdict instead of being overwritten.
func fillSkippedEmpty(results []TaskResult, tasks []Task, from int, reason error) {
	for i := from; i < len(tasks); i++ {
		if results[i].Result != nil {
			continue
		}
		results[i] = TaskResult{
			Name:   tasks[i].Name,
			State:  StateSkipped,
			Result: skippedResult(reason, tasks[i].Cfg),
		}
	}
}
func skipReasonFor(deadline time.Time, hasDeadline bool) error {
	if hasDeadline && time.Now().After(deadline) {
		return ErrBatchDeadline
	}
	return ErrBatchCanceled
}
