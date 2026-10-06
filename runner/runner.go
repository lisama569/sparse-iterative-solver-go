package runner

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/solver"
	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// Task is one independent linear system to solve. A and B are read-only
// inputs and may be shared by any number of tasks.
type Task struct {
	Name string
	A    *sparse.CSR
	B    []float64
	Cfg  solver.Config
}

// TaskResult is the outcome of a Task. A solver-level failure (invalid
// input, breakdown, ...) is reported through Result/Err; the pool itself
// never panics on a bad task.
type TaskResult struct {
	Name   string
	Result *solver.Result
	Err    error
	// State distinguishes fresh computation, reuse across batch resumes and
	// tasks that never started (skipped due to batch limits).
	State TaskState
	// Duration is the wall-clock time spent computing in this batch run
	// (zero for reused/skipped tasks).
	Duration time.Duration
}

// entry is one unit of internal work.
type entry struct {
	task Task
	ctx  context.Context
	done chan *TaskResult
	// hold, when non-nil, makes the worker wait for it before processing
	// the entry. It exists to make queue/backpressure tests deterministic.
	hold chan struct{}
}

// Pool runs tasks with a fixed worker count and a bounded pending queue.
// Memory is therefore bounded by O(queue + workers) task footprints
// regardless of how many goroutines call Submit.
type Pool struct {
	jobs   chan *entry
	mu     sync.Mutex
	closed bool
	wg     sync.WaitGroup
}

// ErrPoolConfig is returned for nonsensical pool parameters.
var ErrPoolConfig = errors.New("pool configuration error")

// ErrPoolStopped is returned after Shutdown (or from a Get whose pool
// stopped before the result could be produced).
var ErrPoolStopped = errors.New("pool stopped")

// NewPool creates a pool with workers concurrent solvers and a queue that
// holds at most queueCapacity pending tasks. Both must be positive.
func NewPool(workers, queueCapacity int) (*Pool, error) {
	if workers <= 0 {
		return nil, fmt.Errorf("%w: workers must be positive, got %d", ErrPoolConfig, workers)
	}
	if queueCapacity <= 0 {
		return nil, fmt.Errorf("%w: queue capacity must be positive, got %d", ErrPoolConfig, queueCapacity)
	}
	p := &Pool{jobs: make(chan *entry, queueCapacity)}
	for i := 0; i < workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}
	return p, nil
}

func (p *Pool) worker(id int) {
	defer p.wg.Done()
	for e := range p.jobs {
		if e.hold != nil {
			<-e.hold
		}
		// A canceled context must not stop the worker: skip the task and
		// report cancellation so other tasks are unaffected.
		if err := e.ctx.Err(); err != nil {
			p.finish(e, &TaskResult{
				Name:   e.task.Name,
				Result: &solver.Result{Status: solver.StatusCanceled, ErrorMessage: err.Error()},
			})
			continue
		}
		r, err := solver.Solve(e.ctx, e.task.A, e.task.B, e.task.Cfg)
		p.finish(e, &TaskResult{Name: e.task.Name, Result: r, Err: err})
	}
}

func (p *Pool) finish(e *entry, tr *TaskResult) {
	if tr.Result == nil && tr.Err == nil {
		tr.Err = errors.New("solver returned no result")
	}
	// The buffered channel guarantees the worker never blocks even if
	// nobody calls Get.
	e.done <- tr
}

// Future is a handle to a submitted task.
type Future struct {
	res chan *TaskResult
}

// Submit enqueues a task and returns a Future. It applies backpressure by
// blocking when the queue is full; ctx allows abandoning the wait without
// affecting already-queued work.
func (p *Pool) Submit(ctx context.Context, t Task) (*Future, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrPoolStopped
	}
	p.mu.Unlock()

	ch := make(chan *TaskResult, 1)
	e := &entry{task: t, ctx: ctx, done: ch}
	// Prefer an immediate, non-blocking enqueue so a task that merely has
	// an already-canceled context is still accepted (the worker reports
	// the cancellation). Only when the bounded queue is full do we block
	// and allow the caller to abandon the wait via ctx.
	select {
	case p.jobs <- e:
	default:
		select {
		case p.jobs <- e:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &Future{res: ch}, nil
}

// Get blocks until the task finished, ctx is canceled or the pool stops.
// Canceling Get only stops waiting; the task itself keeps running unless
// its own Submit context was canceled.
func (f *Future) Get(ctx context.Context) (*TaskResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case tr := <-f.res:
		return tr, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Shutdown stops accepting work and waits for in-flight tasks to finish.
// Pending queued tasks are allowed to complete as well (workers drain the
// bounded queue); callers that want to abandon work should cancel the
// contexts they passed to Submit before calling Shutdown.
func (p *Pool) Shutdown() {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	close(p.jobs)
	p.mu.Unlock()
	p.wg.Wait()
}

// RunAll submits tasks and collects every result in input order. A failure
// or cancellation of one task never removes or alters the others.
func RunAll(ctx context.Context, workers int, tasks []Task) ([]TaskResult, error) {
	pool, err := NewPool(workers, len(tasks)+1)
	if err != nil {
		return nil, err
	}
	defer pool.Shutdown()
	results := make([]TaskResult, len(tasks))
	futures := make([]*Future, len(tasks))
	for i, t := range tasks {
		f, err := pool.Submit(ctx, t)
		if err != nil {
			results[i] = TaskResult{Name: t.Name,
				Result: &solver.Result{Status: solver.StatusInvalidInput, ErrorMessage: err.Error()},
				Err:    err}
			continue
		}
		futures[i] = f
	}
	for i, f := range futures {
		if f == nil {
			continue
		}
		tr, err := f.Get(ctx)
		if err != nil {
			results[i] = TaskResult{Name: tasks[i].Name,
				Result: &solver.Result{Status: solver.StatusCanceled, ErrorMessage: err.Error()},
				Err:    err}
			continue
		}
		results[i] = *tr
	}
	return results, nil
}
