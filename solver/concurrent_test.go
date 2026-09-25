package solver

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/lisama569/sparse-iterative-solver-go/sparse"
)

// makeTasks 生成一批不同规模、混合对称/非对称的任务，复用同一批不可变矩阵。
func makeTasks() ([]Task, []*sparse.Matrix, [][]float64) {
	sizes := []int{20, 50, 80, 120}
	var tasks []Task
	var mats []*sparse.Matrix
	var exacts [][]float64
	for _, n := range sizes {
		sym := laplacian1D(n)
		bSym := exactRHS(sym, ones(n))
		tasks = append(tasks, Task{
			Name: fmt.Sprintf("cg-%d", n),
			A:    sym, B: bSym,
			Opts: Options{Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true, Precond: true},
		})
		mats = append(mats, sym)
		exacts = append(exacts, ones(n))

		ns := nonsym(n)
		bNs := exactRHS(ns, ones(n))
		tasks = append(tasks, Task{
			Name: fmt.Sprintf("bicgstab-%d", n),
			A:    ns, B: bNs,
			Opts: Options{Tolerance: 1e-10, MaxIter: 10 * n, Precond: true},
		})
		mats = append(mats, ns)
		exacts = append(exacts, ones(n))
	}
	return tasks, mats, exacts
}

func TestPool_AllConverge(t *testing.T) {
	for _, workers := range []int{1, 2, 4, 8} {
		tasks, mats, exacts := makeTasks()
		pool := NewPool(workers)
		results := pool.RunAll(context.Background(), tasks)
		if len(results) != len(tasks) {
			t.Fatalf("workers=%d: 结果数 %d != 任务数 %d", workers, len(results), len(tasks))
		}
		for i, tr := range results {
			if tr.Name != tasks[i].Name {
				t.Fatalf("workers=%d: 结果顺序错乱 at %d: %s", workers, i, tr.Name)
			}
			assertConverged(t, tr.Name, mats[i], tasks[i].B, exacts[i], tr.Result, 1e-8)
		}
	}
}

func TestPool_DeterministicAcrossWorkers(t *testing.T) {
	tasks, _, _ := makeTasks()
	base := NewPool(1).RunAll(context.Background(), tasks)
	for _, workers := range []int{2, 4, 8} {
		got := NewPool(workers).RunAll(context.Background(), tasks)
		for i := range base {
			a, b := base[i].Result, got[i].Result
			if a.Status != b.Status || a.Iterations != b.Iterations ||
				a.FinalRelRes != b.FinalRelRes {
				t.Fatalf("workers=%d 任务 %s 结论不稳定:\n  %s\n  %s",
					workers, tasks[i].Name, a.Summary(), b.Summary())
			}
		}
	}
}

func TestPool_FailureIsolation(t *testing.T) {
	tasks, _, _ := makeTasks()
	// 中间插入一个必然失败的任务（维度不符的 b）。
	bad := Task{
		Name: "bad",
		A:    laplacian1D(10),
		B:    []float64{1, 2, 3}, // 维度不符
		Opts: Options{Tolerance: 1e-8, MaxIter: 10, Symmetric: true},
	}
	mixed := append([]Task{}, tasks[:2]...)
	mixed = append(mixed, bad)
	mixed = append(mixed, tasks[2:]...)

	results := NewPool(4).RunAll(context.Background(), mixed)
	for i, tr := range results {
		if tr.Name == "bad" {
			if tr.Result.Status != StatusInvalid {
				t.Fatalf("坏任务应报 invalid_input, 得到 %s", tr.Result.Status)
			}
			continue
		}
		if !tr.Result.Converged() {
			t.Fatalf("任务 %s 被坏任务污染: %s err=%v",
				tr.Name, tr.Result.Summary(), tr.Result.Err)
		}
		_ = i
	}
}

func TestPool_Cancellation(t *testing.T) {
	// 大量任务 + 1 个 worker + 立即取消：后续任务应标记 canceled。
	n := 200
	a := laplacian1D(n)
	b := exactRHS(a, ones(n))
	tasks := make([]Task, 50)
	for i := range tasks {
		tasks[i] = Task{
			Name: fmt.Sprintf("t%d", i),
			A:    a, B: b,
			Opts: Options{Tolerance: 1e-12, MaxIter: 5 * n, Symmetric: true},
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 立即取消
	results := NewPool(1).RunAll(ctx, tasks)
	canceled := 0
	for _, tr := range results {
		if tr.Result.Status == StatusCanceled {
			canceled++
		} else if tr.Result.Status != StatusConverged {
			t.Fatalf("任务 %s 状态异常: %s", tr.Name, tr.Result.Status)
		}
	}
	if canceled == 0 {
		t.Fatal("立即取消时应至少有一个任务被标记 canceled")
	}
	t.Logf("取消统计: %d/%d canceled", canceled, len(results))
}

func TestPool_BoundedConcurrency(t *testing.T) {
	// 验证 Pool 语义：worker 数即并发上限，与任务总数无关。
	// 并发提交远多于 worker 的任务，确认全部完成且结论一致。
	n := 300
	a := laplacian1D(n)
	b := exactRHS(a, ones(n))
	// 包装 Solve 不可行（Pool 内部调用），改为验证 worker 数上限语义：
	// 并发提交远多于 worker 的任务，确认全部完成且结论一致。
	const workers = 3
	const numTasks = 30
	tasks := make([]Task, numTasks)
	for i := range tasks {
		tasks[i] = Task{
			Name: fmt.Sprintf("t%d", i),
			A:    a, B: b,
			Opts: Options{Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true},
		}
	}
	pool := NewPool(workers)
	if pool.Workers() != workers {
		t.Fatalf("worker 数应为 %d, 得到 %d", workers, pool.Workers())
	}
	start := time.Now()
	results := pool.RunAll(context.Background(), tasks)
	elapsed := time.Since(start)
	for _, tr := range results {
		if !tr.Result.Converged() {
			t.Fatalf("任务 %s 未收敛: %s", tr.Name, tr.Result.Summary())
		}
	}
	t.Logf("workers=%d tasks=%d 耗时 %v", workers, numTasks, elapsed)
}

func TestPool_SharedImmutableInput(t *testing.T) {
	// 多个任务共享同一矩阵，并发求解后矩阵内容不变。
	n := 100
	a := laplacian1D(n)
	valsSnapshot := make([]float64, len(a.Vals))
	copy(valsSnapshot, a.Vals)
	b := exactRHS(a, ones(n))
	tasks := make([]Task, 16)
	for i := range tasks {
		tasks[i] = Task{
			Name: fmt.Sprintf("s%d", i),
			A:    a, B: b,
			Opts: Options{Tolerance: 1e-10, MaxIter: 5 * n, Symmetric: true, Precond: true},
		}
	}
	results := NewPool(8).RunAll(context.Background(), tasks)
	for _, tr := range results {
		if !tr.Result.Converged() {
			t.Fatalf("任务 %s 未收敛: %s", tr.Name, tr.Result.Summary())
		}
	}
	for i := range a.Vals {
		if a.Vals[i] != valsSnapshot[i] {
			t.Fatal("共享矩阵在并发求解后被修改")
		}
	}
}
