# sparse-iterative-solver-go

一套可完全本地运行的稀疏线性系统迭代求解组件。支持从 **Matrix Market 文件**
或**坐标三元组**构造矩阵，提供面向**对称正定系统**与**一般系统**的迭代求解器
（可选 Jacobi 预条件），并通过有界并发池安全地同时执行大量独立求解任务。

不依赖任何网络服务、凭据或付费资源；只使用 Go 标准库。

## 包结构

| 包 | 职责 |
| --- | --- |
| `sparse` | 坐标矩阵（COO）、压缩稀疏行（CSR）、校验与错误分类、矩阵-向量乘 |
| `mmio` | Matrix Market `coordinate` 格式解析（`real` / `integer` / `pattern`，`general` / `symmetric`） |
| `solver` | 迭代求解：CG 与重启 GMRES(m)，可选 Jacobi 预条件，统一的状态判定、可归档判定记录、分段续算检查点 |
| `runner` | 固定 worker 数、有界队列的任务池（取消、失败隔离、背压），以及带共享预算与续跑语义的整批执行 |
| `cmd/solver` | 本地命令行入口 |

## 支持的数据形式

1. **坐标三元组**：直接构造 `sparse.COO{Rows, Cols, Triplets, Symmetric}`。
   索引为 0 基。`Symmetric=true` 时只需提供一个三角（含可选对角），转换为 CSR
   时会对称展开。
2. **Matrix Market 文件**：`mmio.ReadFile(path)` 或 `mmio.ReadCOO(reader)`。
   文件内索引为格式规定的 1 基，解析时自动转换。支持
   `%%MatrixMarket matrix coordinate {real|integer|pattern} {general|symmetric}`，
   允许以 `%` 开头的注释行。`pattern` 字段的非零元取值一律为 `1.0`。

两种入口都会执行完整校验：

- 维度必须为正；对称矩阵必须为方阵；
- 索引必须落在声明维度之内；
- **重复坐标一律拒绝**（对称输入中同时给出 `(i,j)` 与 `(j,i)` 也算重复），
  不会静默求和或丢弃；
- 数值必须有限（拒绝 `NaN`/`+Inf`/`-Inf`）；
- 文件中声明的非零元数量必须与实际数据行一致。

## 求解路径与适用条件

`Config.Method`：

- `MethodAuto`（默认）：矩阵元数据 `Symmetric=true` 时选择 **CG**，否则选择 **GMRES**。
- `MethodCG`：共轭梯度法，适用于**对称正定（SPD）方阵**。显式对非对称矩阵选择
  CG 会在求解前以结构错误拒绝（`MethodAuto` 不受影响，会自动改走 GMRES）。
  对对称但非正定的矩阵，当出现非正曲率 `pᵀAp ≤ 0` 时报告 `StatusBreakdown`，
  **绝不会**把这类结果标记为收敛。
- `MethodGMRES`：重启 GMRES(m)（修改 Gram–Schmidt Arnoldi + Givens 旋转），
  对一般方阵不做对称性假设，可处理非对称和非正定但非奇异的系统。重启长度由
  `Config.Restart` 控制，默认 `min(n, 50)`。

其他配置：`Tolerance`（相对残差阈值 `‖r‖/‖b‖`，默认 1e-8）、`MaxIterations`
（矩阵-向量乘步数上限，默认 `10n`）、`InitialGuess`（nil 表示零向量）、
`Precondition`（Jacobi 对角预条件）。

Jacobi 预条件要求对角元全部非零且有限，否则在求解前以数值类错误拒绝。

### 结果状态（严格区分，互不混用）

| 状态 | 含义 |
| --- | --- |
| `StatusConverged` | 实测相对残差 `≤ Tolerance`。收敛判定基于真实/最小二乘残差，入口处还有最终一致性校验 |
| `StatusMaxIterations` | 步数用尽但残差仍高于容差 |
| `StatusBreakdown` | 算法失效（CG 非正曲率、GMRES Krylov 基崩溃、出现非有限值） |
| `StatusInvalidInput` | 矩阵/右端项/配置不合法，同时返回带分类的非空 `error` |
| `StatusCanceled` | 求解过程中上下文被取消（分段求解时会带回可续算的检查点） |
| `StatusPaused` | 分段预算用完、尚未收敛也未失败；检查点可续算，**不是**失败 |
| `StatusSkipped` | 整批预算耗尽时任务尚未开始；任务从未被尝试，**不是**求解失败 |

**任何未满足残差要求的结果都不会被标记为成功**：GMRES 重启后、步数耗尽后都会
重新计算真实残差，`Solve` 末尾还有 `rel ≤ tol` 的兜底断言。

每次求解返回 `History`（初始相对残差与每步相对残差），供日志与可复现核查。

### 判定记录（`Result.Record()`）

每个 `Result` 都可以导出一份可归档的判定记录 `solver.Record`（带 JSON 标签，
可直接编进批处理汇总）：

| 字段 | 含义与读法 |
| --- | --- |
| `status` | 机读判定：`converged` / `max_iterations` / `breakdown` / `invalid_input` / `canceled` / `paused` / `skipped` |
| `converged` | 只有「状态为 converged **且**实测残差 ≤ 容差」才为 true；被篡改或不一致的成功标记会被拒绝 |
| `iterations` | 实际推进的矩阵-向量乘步数（分段求解时为跨段累计值） |
| `final_rel_res` | 最终实测相对残差 `‖r‖/‖b‖` |
| `tolerance` | 判定所用的容差（已归一化默认值） |
| `history` | 收敛走势：初始相对残差与每一步后的相对残差 |
| `reason` | 人读的判定依据（含残差、容差、步数与失败细节） |

### 分段续算（`SolveSegment` + `Checkpoint`）

`solver.SolveSegment(ctx, A, b, cfg, prev, segmentBudget)` 把一次求解拆成若干段：

- `prev == nil` 从头开始（尊重 `cfg.InitialGuess`）；否则从上一段返回的
  `Checkpoint` 续算。返回的 `Result` 报告的是**跨段累计**的步数与收敛走势。
- 段预算用完且尚未收敛/失败时返回 `StatusPaused` 与非空检查点；中途被取消时
  返回 `StatusCanceled` 与保存了已完成进展的检查点。终止状态（收敛、步数用尽、
  失效、输入非法）返回空检查点。
- `cfg.MaxIterations` 是**跨段累计**的总步数上限；总上限耗尽返回
  `StatusMaxIterations`（终止，不可续算）。
- **一致性保证**：分段续算与一口气跑完的轨迹完全相同——收敛与否、总步数、
  最终残差、收敛走势、解向量逐位一致（测试 `TestSegmentedMatchesOneShot` 覆盖
  CG/GMRES × 是否预条件 × 多种段长）。CG 在任意迭代处切段；GMRES 在重启边界
  切段，因此单段实际步数最多超出段预算一个重启周期（绝不超出总上限）。
- 检查点可 JSON 序列化，可落盘后跨进程交接。续算时校验维度、算法、容差、
  重启长度与预条件设置，不匹配会报错并**原样退回**检查点，已算出的进展
  不会被破坏，没算完的状态也不会被当成完成。

## 错误分类

输入类错误统一为 `*sparse.InputError`，带可程序化判别的 `Kind`：

- `sparse.KindFormat`：文本无法按所声明格式解析（缺少 banner、字段数不对、
  数字无法解析、不支持的字段/对称性等）；
- `sparse.KindStructure`：结构冲突（维度非正、非方阵对称、越界索引、重复项、
  声明条目数不符、向量长度不匹配等）；
- `sparse.KindNumeric`：数值不可用（`NaN`/`Inf`、Jacobi 对角元为零或非有限）。

文件打开失败等 I/O 错误保持普通 `error`（不伪装成矩阵错误）。调用方可用
`errors.As(err, &inputErr)` 区分三类原因，组件从不静默修正输入。

## 并发与资源约束

`runner.NewPool(workers, queueCapacity)` 创建固定数量的 worker goroutine 和
容量固定的待办队列：

- goroutine 数量恒为 `workers`，**不会随并发提交数增长**；
- 待办内存占用上限为 `O(queueCapacity + workers)`；队列满时 `Submit` 阻塞背压，
  也可通过传入的 `context.Context` 放弃等待；
- 矩阵 `*sparse.CSR` 和右端项是**只读共享**的，多个任务复用同一份不可变输入，
  不需要拷贝（所有测试在 `-race` 下通过）；
- 每个任务携带自己的 `context`，取消一个任务或一个任务失败（非法输入、
  不收敛、breakdown）都只影响它自己，worker 继续处理其余任务；
- `Future.Get` 的取消只中断等待，不杀死已提交的任务（任务由提交时的 context
  控制）；`Shutdown` 停止接收新任务并等待在途任务结束。
- 求解算法本身无随机性，固定配置、固定输入下重复运行给出逐位一致的结论
  （测试 `TestDeterminism` / `TestDeterministicAcrossRuns` 覆盖）。

## 整批预算与续跑（`runner.RunBatch`）

`runner.RunBatch(ctx, opts, tasks, prev)` 在固定 worker 数下跑整批任务，
结果**始终按提交顺序返回**，每个任务恰有一份结果，不会悄悄消失。

`BatchOptions.Budget` 给整批设共享上限（零值表示不限）：

- `MaxTotalIterations`：整批所有任务合计的迭代步数上限。`SegmentSize > 0` 时
  任务按段推进、段间检查预算，预算耗尽时未开始的任务标记 `StatusSkipped`、
  算了一半的任务带检查点暂停（`StatusPaused`）；`SegmentSize <= 0` 时每个任务
  单段跑完，预算只在任务之间检查。
- `MaxElapsed`：整批墙钟上限。到期时在途任务被取消（检查点保留进展）、
  未开始的任务标记 `StatusSkipped`。

续跑：把上一次返回的结果切片作为 `prev` 再调 `RunBatch`（任务列表需与上次
按位置一一对应）。已达终止状态（收敛/步数用尽/失效/输入非法）的任务**原样
保留、绝不重算**；暂停或取消的任务从检查点继续；跳过的任务重新开始。某个
任务超预算、被跳过或被取消都只影响它自己，其他任务的结论不受影响。

每份 `BatchResult` 可通过 `Record()` 导出判定记录，整批记录可直接
`json.Marshal` 进汇总文件。

## 本地验证

要求 Go 1.25+，无需联网（无第三方依赖）。

```sh
go build ./...
go vet ./...
go test ./...            # 全部单元/并发测试
go test -race ./...      # 竞态检测
go test ./... -v         # 查看任务级日志：输入规模、收敛历史摘要、判定依据
go test ./solver/ -run 'Segment|Record' -v   # 分段续算一致性与判定记录
go test ./runner/ -run Batch -v              # 整批预算、跳过与续跑
```

命令行使用：

```sh
go run ./cmd/solver \
  -matrix path/to/A.mtx [-rhs b.txt] [-x0 x0.txt] \
  -method auto|cg|gmres [-tol 1e-8] [-maxiter N] \
  [-restart M] [-precond] [-history]
```

退出码：`0` 收敛；`1` 达到迭代上限或算法失效；`2` 输入/文件错误。

### 作为库使用

```go
coo, err := mmio.ReadFile("A.mtx")          // 或直接构造 sparse.COO
A, err := coo.ToCSR()
res, err := solver.Solve(ctx, A, b, solver.Config{
    Tolerance: 1e-10, Precondition: true,
})
switch res.Status {
case solver.StatusConverged: // res.X 可用，res.FinalRelRes ≤ tol
case solver.StatusMaxIterations, solver.StatusBreakdown: // res.History 说明原因
}
```

## 测试覆盖

- 畸形矩阵：坏 banner、不支持的字段/对称性、尺寸行错误、条目数不符、越界索引、
  重复项（含对称对）、非方阵对称、`NaN`/`Inf`；
- 求解异常：非正定矩阵上的 CG breakdown、容差过严/步数过少导致的 max iterations、
  非对称矩阵走 GMRES、零初值残差即收敛；
- 并发：多种规模（2–128）与 worker 数（1/2/4/8）、任务失败与取消的隔离、
  队列背压、Shutdown 后拒绝提交、结果确定性；
- 分段续算：CG/GMRES（含预条件）多种段长下与一次性求解的结论逐位一致、
  总步数上限跨段累计、取消后检查点保留进展并可续算、检查点交接不匹配时
  报错且原样退回、已知真解场景下续算结果既满足残差要求又逼近真值；
- 判定记录：收敛/步数用尽/失效/输入非法/取消/暂停/跳过七种结束方式的记录
  完整且互相可区分，伪造的成功标记会被拒绝；
- 整批：迭代预算耗尽时未开始任务明确标记为跳过、续跑不重算已完成任务、
  时间预算到期可续跑、单任务失败/取消不影响其他任务、结果按提交顺序返回；
- 断言以可复现的相对残差和误差为准；失败日志包含任务名、输入规模、
  收敛历史摘要（首项/末项/长度）与判定依据。
