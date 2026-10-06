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
| `solver` | 迭代求解：CG 与重启 GMRES(m)，可选 Jacobi 预条件，统一状态判定、**可归档判定记录**、**分段续算** |
| `runner` | 固定 worker 数、有界队列的任务池，支持取消、失败隔离、背压；**通宵整批作业（总预算、跳过、跨次续跑）** |
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
| `StatusCanceled` | 求解过程中上下文被取消，**已算出的进度随结果完整返回** |
| `StatusPaused` | 一段的步数上限到了但还没到终态；`Result.Progress` 可交给下一段继续，**不是成功** |
| `StatusSkipped` | 整批运行中任务**还没轮到/还没开始**，批预算或时间已耗尽（或批次 context 先取消）；**不是求解失败** |

**任何未满足残差要求的结果都不会被标记为成功**：GMRES 重启后、步数耗尽后都会
重新计算真实残差，`Solve` 末尾还有 `rel ≤ tol` 的兜底断言。

每次求解返回 `History`（初始相对残差与每步相对残差），供日志与可复现核查。

## 判定记录（可归档）

`(*solver.Result).Report()`（批量场景用 `runner.TaskResult.Record()` 自动带上任务名）
返回一份不含大向量、可直接 `json.Marshal` 进批处理汇总的 `solver.Report`。
**任何结束方式**（收敛 / 步数用满 / 算法失效 / 输入不合法 / 中途取消 / 分段暂停 /
整批跳过）都能拿到完整记录。字段读法：

| 字段 | 含义 |
| --- | --- |
| `name` | 整批任务名（单次求解为空） |
| `method` | 实际使用的算法：`cg` / `gmres` |
| `status` | 机读结论：`converged` / `max_iterations` / `breakdown` / `invalid_input` / `canceled` / `paused` / `skipped` |
| `success` | **仅当**实测残差达标且状态为 converged 才为 true；其余任何情况都为 false |
| `iterations` | 实际推进的矩阵-向量乘总步数 |
| `segment_iterations` | 最近一段（或本次批量运行）实际推进的步数；跨次复用的旧任务为 0 |
| `tolerance` | 本次实际应用的相对残差阈值 |
| `final_rel_res` | 最终**实测**相对残差 `‖r‖/‖b‖`（`‖b‖=0` 时为绝对残差） |
| `residual_measured` | 残差是否可信；输入不合法或算出非有限值时为 false |
| `residual_met` | 是否 `final_rel_res ≤ tolerance`（先要求 measured 为 true） |
| `history` | 完整残差走势：第 0 项为初始残差，之后每步一项；GMRES 还含重启/分段边界的显式实测点 |
| `basis` | 判定成该结论的**依据**（带测量值与阈值的一句话），例如 `measured relative residual 8.3e-11 <= tolerance 1e-10 after 27 iteration(s)` |
| `detail` | 失效/非法/取消/跳过的具体原因；收敛时为空 |

硬性不变量：`success=true` 当且仅当残差确实达标；GMRES 收敛、重启后、步数耗尽后都
重新计算真实残差，入口处还有 `rel ≤ tol` 的兜底断言——**没真正满足残差要求的结果
永远不会看起来是成功**。

## 分段续算（checkpoint / resume）

一个求解可以拆成几段接着算：

```go
// 第一段：最多推进 50 步，到上限就停下并返回可续算进度
r1, _ := solver.Advance(ctx, A, b, cfg, nil, 50)
// r1.Status == solver.StatusPaused, r1.Progress 是非空的不透明检查点

// 之后拿着进度继续：再推一段（0 表示“用完 cfg.MaxIterations 剩余预算”）
r2, _ := solver.Advance(ctx2, A, b, cfg, r1.Progress, 0)
```

- `segmentSteps` 是**这一段**的步数上限；`cfg.MaxIterations` 是**总**步数上限，
  多段步数之和不会超过它。
- 到段上限且尚未收敛 → `StatusPaused` + 非空 `Progress`；收敛/用满/失效为终态且
  `Progress == nil`。
- **中途取消也不丢进度**：段内 context 被取消时返回 `StatusCanceled` + 非空
  `Progress`（记录里仍有已走步数、实测残差与走势），换个 context 可以无损继续。
- 续拼出来的结论与一口气跑完一致：收敛与否、总步数、最终残差对得上；CG 逐位一致，
  GMRES 仅在边界测量上允许浮点层面的极小差异。GMRES 检查点会保留**周期内**的
  Arnoldi 状态（不是提前重启），因此分段不损失 Krylov 信息。
- 交接校验：续算的维度、方法、容差、预条件、重启长度、总预算必须与产生进度时一致，
  否则返回 `StatusInvalidInput` + `solver.ErrProgressMismatch`；被拒的续算**不会**
  修改传入的进度（内部对检查点做深拷贝，旧快照不受后续段影响）。
- 一次性 `solver.Solve` 等价于 `Advance(ctx, A, b, cfg, nil, 0)`。

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

## 通宵批量作业：总预算、跳过与跨次续跑

`runner.RunBatch(ctx, runner.BatchRequest{Tasks, Cfg, Previous})` 面向一整夜的大批量作业：

```go
sum, err := runner.RunBatch(ctx, runner.BatchRequest{
    Tasks: tasks, // 严格按提交顺序
    Cfg: runner.BatchConfig{
        Workers:             4,
        SegmentSteps:        64,    // 每段最多 64 步，预算/超时检查粒度
        MaxTotalIterations:  2e6,  // 全批共享的总矩阵-向量乘步数上限（<=0 不限）
        MaxDuration:         8 * time.Hour, // 全批总墙钟时间（<=0 不限）
    },
    Previous: lastNight.Results, // 可选：拿上一次结果续跑整批
})
for _, rec := range sum.Records() { // []solver.Report，可直接写汇总/JSON
    ...
}
```

语义（均有测试锁定）：

- **严格按提交顺序返回**：`sum.Results[i]` 永远对应 `Tasks[i]`，并发不改变这个顺序。
- **总工作量预算 `MaxTotalIterations`**：所有任务共享、按实际推进的矩阵-向量乘步数累计
  （`sum.IterationsUsed`）。预算在「在途任务」内用完时，该任务停在当前段并以
  `StatusPaused`（带进度）返回；**还没轮到、还没开始的任务一律标为
  `StatusSkipped`**，记录里写明原因（预算耗尽 / 时间耗尽 / 批次 context 取消），
  既不会悄悄消失，也不会被当成求解失败或假成功（步数 0、残差标记为不可用）。
- **总时间预算 `MaxDuration`**：到点后在途任务保留进度结束（`StatusPaused` /
  `StatusCanceled` + 非空进度），未启动任务标为 `StatusSkipped`（原因=时间耗尽）。
  调用方自己的 `ctx` 取消则原因记为批次取消；三种原因互相可区分。
- **跨次续跑不重算**：传入 `Previous`（长度/顺序必须与 Tasks 一致，否则报错）时——
  - 已有**终态结论**（收敛 / 步数用满 / 失效 / 非法输入）的任务直接复用同一条记录，
    `State == StateReused`，**不会重新算**，也不消耗本次预算（`Duration` 为 0）；
  - 上次**暂停/取消**的任务拿着 `Progress` 从断点继续，未启动的 `skipped` 任务本次
    作为新任务从头开始；
  - 续跑拼出的收敛结论、总步数、最终残差与一次性跑批一致。
- **任务隔离**：某个任务超预算、被跳过、被取消、输入非法或算法失效，都只影响它自己；
  每个任务独立切片循环，worker 数固定。`sum` 另给 `Converged/Failed/Skipped/Paused`
  计数，便于通宵作业后快速汇总。

`TaskResult.State` 区分 `computed`（本次实际计算）/ `reused`（复用上次终态）/
`skipped`（本次未启动）。单次接口 `runner.RunAll` 与 `Pool` 行为保持不变。

## 本地验证

要求 Go 1.25+，无需联网（无第三方依赖）。

```sh
go build ./...
go vet ./...
go test ./...            # 全部单元/并发/预算/续算测试
go test -race ./...      # 竞态检测
go test ./... -v         # 查看任务级日志：输入规模、收敛历史摘要、判定依据

# 只看新增能力
go test ./solver/ -v -run TestSegment          # 分段续算与一次性等价、取消续算、交接校验
go test ./solver/ -v -run TestReport           # 各结束方式判定记录完整可区分
go test ./runner/ -v -run TestRunBatch         # 整批预算/跳过/续跑不重算/隔离/归档
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

// 归档判定记录
raw, _ := json.MarshalIndent(res.Report(), "", "  ")

// 分段续算：段上限到点 -> paused + progress；继续推进直到终态
r1, _ := solver.Advance(ctx, A, b, cfg, nil, 50)
r2, _ := solver.Advance(ctx, A, b, cfg, r1.Progress, 0)

// 通宵整批：共享预算 + 跨次续跑
sum, _ := runner.RunBatch(ctx, runner.BatchRequest{
    Tasks: tasks,
    Cfg:   runner.BatchConfig{Workers: 4, MaxTotalIterations: 2e6, MaxDuration: 8 * time.Hour},
    // Previous: 上次的 sum.Results —— 终态复用、暂停续算、跳过重跑
})
```

## 测试覆盖

- 畸形矩阵：坏 banner、不支持的字段/对称性、尺寸行错误、条目数不符、越界索引、
  重复项（含对称对）、非方阵对称、`NaN`/`Inf`；
- 求解异常：非正定矩阵上的 CG breakdown、容差过严/步数过少导致的 max iterations、
  非对称矩阵走 GMRES、零初值残差即收敛；
- 并发：多种规模（2–128）与 worker 数（1/2/4/8）、任务失败与取消的隔离、
  队列背压、Shutdown 后拒绝提交、结果确定性；
- 判定记录：收敛 / 步数用满 / 失效 / 非法输入 / 取消 / 暂停每种结论都产出完整、
  相互可区分的 `Report`，且非达标结果的 `success` 一律为 false，可 JSON 归档；
- 分段续算：CG 与 GMRES（含预条件、含周期中段、跨重启边界）多种分段切分下，
  收敛结论 / 总步数 / 最终残差 / 残差走势与一次性求解一致（CG 逐位、GMRES 浮点级）；
  对有已知真解的制造右端项场景，续算后的解既满足残差要求也对得上真值；段中取消可
  无损续算；错误交接返回 `ErrProgressMismatch` 且不破坏已有进度；
- 整批：共享迭代/时间预算耗尽时未启动任务明确标 `skipped` 且原因可区分、在途任务保留
  进度；结果严格按提交顺序返回；拿上次结果续跑时终态任务复用不重算、暂停任务从断点
  继续、跳过任务重新开始；单任务超预算/跳过/取消不污染其他任务；汇总记录可 JSON 归档；
- 断言以可复现的相对残差和误差为准；失败日志包含任务名、输入规模、
  收敛历史摘要（首项/末项/长度）与判定依据。
