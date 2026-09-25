# sparse-iterative-solver-go

本地可运行的稀疏线性系统求解组件：从本地文件加载稀疏矩阵，用迭代法求解
`A*x = b`，支持并发执行多个求解任务。仅依赖 Go 标准库，不访问网络。

## 结构

- `sparse/` — CSR 稀疏矩阵、坐标三元组（COO）与 Matrix Market 格式加载、校验
- `solver/` — CG / BiCGSTAB 迭代求解、Jacobi 预条件、有界并发执行器

## 支持的数据形式

两种本地输入，均为 1-based 索引（内部转为 0-based CSR）：

1. **Matrix Market 坐标格式**（`sparse.LoadMatrixMarket`）
   - 支持 `real` / `integer` 数值，`general` / `symmetric` 结构
   - `symmetric` 输入自动按镜像展开为完整矩阵
   - `complex` / `pattern` / `hermitian` / `skew` 明确拒绝
2. **纯文本三元组**（`sparse.LoadCOO`）
   - 首行 `rows cols nnz`，其后每行 `i j v`，支持 `#` 注释

加载时的校验（不做任何静默修正）：

| 错误 | 触发条件 |
|---|---|
| `sparse.ErrFormat` | 头部/尺寸行/数据行畸形、声明 nnz 与实际不符、不支持的类型 |
| `sparse.ErrDimension` | 维度非正、矩阵与向量维度不符 |
| `sparse.ErrBadIndex` | 索引越界 |
| `sparse.ErrDuplicate` | 同一 `(i,j)` 出现多次（结构冲突，拒绝合并） |
| `sparse.ErrBadValue` | NaN/Inf、零对角元 |
| `sparse.ErrStructure` | 缺少对角元等结构缺陷 |
| `sparse.ErrNotSquare` / `sparse.ErrAsymmetric` | 非方阵、结构不对称 |

均可用 `errors.Is` 区分。

## 求解路径

`solver.Solve(A, b, Options)`：

| 路径 | 适用条件 | 启用方式 |
|---|---|---|
| CG | 对称正定系统；调用前校验结构对称性 | `Options.Symmetric = true` |
| BiCGSTAB | 一般（含非对称）系统 | `Options.Symmetric = false`（默认） |
| Jacobi 预条件 | 对角线完整且非零 | `Options.Precond = true` |

可配置项：相对残差容差 `Tolerance`（>0）、最大步数 `MaxIter`（>0）、
初值 `InitialX`（可选）。收敛判据为 `‖r‖/‖b‖ ≤ Tolerance`（`b=0` 时用绝对残差）。

### 结论分类（`Result.Status`）

| 状态 | 含义 | 对应错误 |
|---|---|---|
| `StatusConverged` | 残差满足容差 | — |
| `StatusMaxIter` | 达到迭代上限仍未满足容差 | `ErrNotConverged` |
| `StatusBreakdown` | 算法失效（`pᵀAp=0`、`rho=0`、`omega=0`、残差发散等） | `ErrBreakdown` |
| `StatusInvalid` | 输入不合法（维度、容差、步数、对称性等） | `ErrInvalidInput` |
| `StatusCanceled` | 任务被取消 | `ErrCanceled` |

任何未满足残差要求的结果都不会标记为成功（`Result.Converged()` 为 false）。
`Result` 携带每步相对残差历史 `ResHistory`，`Result.Summary()` 输出
状态、步数、最终残差与历史摘要，便于失败归因。

## 并发与资源约束

`solver.NewPool(workers)` 创建固定 worker 数的执行器，`RunAll(ctx, tasks)`：

- 任意时刻最多 `workers` 个求解在运行，与任务总数无关；任务通道无缓冲，
  资源占用不随并发提交数增长
- 多个任务可安全共享同一不可变 `*sparse.Matrix` 与右端项（测试中有快照校验）
- 结果顺序与输入一致；单个任务失败不影响其它任务
- `ctx` 取消时，未开始的任务标记 `StatusCanceled`，已完成结果不受影响
- 固定配置下重复运行（含不同 worker 数）得到逐位一致的结论

## 本地验证

```sh
go build ./...
go vet ./...
go test ./...          # 单元测试
go test ./... -race    # 含竞态检测
```

测试覆盖：畸形矩阵（格式/索引/重复/NaN/数量不符）、不收敛与算法失效、
不同规模（n=10…500）与线程配置（1/2/4/8 workers）下的并发正确性、
取消语义、失败隔离、不可变输入复用与结论可复现性。正确性断言均独立
重算残差 `‖b−Ax‖/‖b‖` 与对真解的误差，失败日志打印输入规模、
收敛历史摘要与判定依据。
