package solver

import (
	"context"
	"sync"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// Task 描述一个求解任务；Matrix 与 B 在任务生命周期内必须视为不可变。
type Task struct {
	Name string
	A    *sparse.Matrix
	B    []float64
	Opts Options
}

// TaskResult 携带任务名与求解结果。
type TaskResult struct {
	Name   string
	Result *Result
}

// Pool 以固定数量的 worker 并发执行求解任务。
// 资源占用有界：任意时刻最多 workers 个求解在运行，
// 与提交的任务总数无关。
type Pool struct {
	workers int
}

// NewPool 创建固定 worker 数的执行器；workers <= 0 时取 1。
func NewPool(workers int) *Pool {
	if workers <= 0 {
		workers = 1
	}
	return &Pool{workers: workers}
}

// Workers 返回 worker 数量。
func (p *Pool) Workers() int { return p.workers }

// RunAll 并发执行全部任务，返回与输入顺序一致的结果。
// ctx 取消时未开始的任务标记为 StatusCanceled；已完成与进行中的
// 任务结果不受影响，单个任务失败也不会污染其它任务。
func (p *Pool) RunAll(ctx context.Context, tasks []Task) []TaskResult {
	results := make([]TaskResult, len(tasks))
	for i, t := range tasks {
		results[i].Name = t.Name
	}

	jobs := make(chan int) // 无缓冲：排队任务数不会额外占用求解资源
	var wg sync.WaitGroup
	for w := 0; w < p.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				t := tasks[i]
				results[i].Result = Solve(t.A, t.B, t.Opts)
			}
		}()
	}

loop:
	for i := range tasks {
		select {
		case <-ctx.Done():
			// 取消：剩余未派发的任务标记为 canceled。
			for j := i; j < len(tasks); j++ {
				results[j].Result = &Result{
					Status:      StatusCanceled,
					Err:         ErrCanceled,
					FinalRelRes: -1,
				}
			}
			break loop
		case jobs <- i:
		}
	}
	close(jobs)
	wg.Wait()
	return results
}
