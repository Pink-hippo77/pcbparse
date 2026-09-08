# 后续优化方向与 Agent 执行指南

更新时间：2026-08-24（Asia/Shanghai）  
适用目标：依据《赛题六 PCB 原理图自动解析落地部署实施手册》，继续把当前仓库推进到高度可用、可复核、可提交演练的竞赛系统。  
文档定位：后续 Agent 的首要交接入口。旧文档中的历史指标不得覆盖本文列出的较新证据；所有“通过”仍以对应文件当前字节与哈希复核为准。

## 1. 接手后的第一原则

本项目已经不是从零搭建阶段。后续 Agent 应优先完成正在进行的训练、候选晋级、50 例端到端和发布证据闭环，不要重做已通过的数据物化、Component/Pin 训练或四模型 Junction 架构。

当前发布结论必须保持为：

- 工程模块基本齐备；
- Component 与 Pin 已有可复核的较新候选；
- OCR 正在正式训练；最新完成 epoch 7 且刷新最佳，Wire 等待 OCR 门禁通过后自动接力；
- Learned Junction 仍不合格，正式架构已改为四个学习模型加确定性 Junction 向量化；
- 主 `models/models.lock.json` 仍是旧的、全部未祝福候选，不能用于生产放行；
- release50/源码/Net H−y/真实 local→global 四项证据绕过已在代码和负向测试中修复，但尚未生成新锁下的正式证据；
- 尚未完成新四模型锁、三次 50 例推理、正式坐标报告、接口联调和目标容器演练；
- 官方 evaluator/sample submission 仍缺失，任何 Agent 都不得声称官方得分或“保证获奖”。

## 2. 不可变事实与数据边界

### 2.1 数据

- 官方 Dataset300 已入库并完成 300 样本、240/30/30 grouped train/val/test 划分。
- 训练 lineage 使用 `dataset300-v2`；源清单、split 和各任务清单 SHA 已绑定在训练配置与终态文件中。
- `val` 可用于模型和阈值选择；内部 `test` 只允许一次性独立确认，不可反向搜索阈值。
- 正式系统演练套件为 `data/splits/dataset300_release50_v1.jsonl`，必须使用 split `release50`、恰好 50 个案例。任何 1 例或自选 manifest 只能算诊断，不能算正式 release50 证据。
- release50 图像根为 `work/release50-images-only`；构建报告为 `artifacts/release/release50_suite_build.json`。

### 2.2 官方边界

- 当前权威参考为 `artifacts/reference/2026_EDA_Challenge_Task6_AI_PCB_Schematic_Official_Guide.pdf`。
- 本地 serializer 和 raw-v1 指标是按现有指南实现的工程口径，不是官方隐藏 evaluator。
- 获得官方样例或 evaluator 后，必须重新确认坐标系、bbox 顺序、空字段、Net HyperGraph、NetLine、文件命名和评分聚合。

### 2.3 发布纪律

- 不得手工把任何候选的 `status` 改为 `blessed`。
- 不得直接编辑 `models/models.lock.json` 冒充晋级。
- promotion 只能消费哈希可复核的候选 bundle，并写到全新的锁和报告路径。
- 正式确定性 Junction 模式下，模型集合必须恰好为 `component/ocr/pin/wire` 四项；不得加载、登记或调用 learned Junction。
- smoke、阈值扫描、单图诊断和旧锁推理均不能替代正式 50 例证据。

## 3. 当前模块真实状态

| 模块 | 当前状态 | 最新证据 | 后续动作 |
|---|---|---|---|
| Dataset/物化 | 已完成 | Dataset300-v2 lineage、五任务物化与 split 均已复核 | 除非数据字节变化，不重物化 |
| Circuit-IR/状态机 | 已实现 | 生产 evidence 已持久化 local geometry、view matrix 与 restored geometry | 用新锁 release50 验证逐对象绑定覆盖 |
| Component | 候选已通过独立 test 与生产适配器 smoke | 新候选见下文 | 保留，不再从验证集重复调参；进入新锁和 E2E |
| Pin | v5 候选已通过正式多门禁和生产 smoke | `work/candidates-runtime-bound/pin-v5-boundary/bundle.manifest.json` | 保留，进入新锁和 E2E |
| OCR | v5 正式训练中 | epoch 7 detector Hmean `0.4100008039`，目标 `0.45`；其余三门通过 | 等待终态；不得提前登记 |
| Wire | v5 等待器正常 | `work/wire-v5-launcher.log` | OCR 四门全过后自动启动 fresh v5 |
| Junction | learned 模型隔离；确定性向量化启用 | 四模型 task-plan/promotion/readiness 定向测试通过 | 不再以六类 learned Junction 阻塞本轮四模型发布 |
| Association/Graph | 工程链路已实现，系统质量待 E2E | Component/OCR/Pin/Wire 输出可进入关联和拓扑 | 用 release50 分层定位 split/merge、缺 key/number、断网 |
| CLI/HTTP | 接口结构与 readiness 已通过定向测试 | CLI/API 定向回归通过 | 新锁后做真实模型接口联调、错误码和并发边界 |
| Release evidence | 四个 P1 绕过点已完成代码修复与负向测试 | 见第 7 节；定向坐标测试 10 项通过；最新硬门禁诊断见 7.5 | 新锁后生成同一 run 绑定的正式证据 |
| Docker/离线 | Dockerfile 存在，当前主机无 Docker 可执行文件 | 尚无目标容器证据 | 新锁/E2E 后在有 Docker 的目标机执行 |

## 4. 已完成且不要重做的优化

### 4.1 Component v5 校准候选

模型权重保持不变，artifact SHA-256：

`34d79a6dfa4340e64dc96de4945bcfea6e376d9a6f4949876b58d7e719697af5`

验证集仅用于选定 score threshold `0.6415358781814575`；随后在未参与阈值选择的 30 张冻结内部 test 图上独立确认：

- mAP50：`0.4628123153`，通过 `>= 0.25` 门禁；
- Precision50：`0.8215802889`；
- Recall50：`0.6701316701`；
- F1@50：`0.7381679389`；
- 相对 score floor `0.20`，F1 提升 `0.0282023032`，mAP50 下降 `0.0386254251`，仍显著高于门禁。

权威文件：

- 校准选择：`artifacts/audits/component_v4_v5_stitched_threshold_sweep.json`
- 独立确认：`work/confirm-component-v5-calibrated-test/calibrated_test_status.json`
- 新候选：`work/candidates-runtime-bound/component-v5-calibrated-test/bundle.manifest.json`
- 候选 bundle SHA-256：`deb737384945d9b8e280e25c98e8067c4cb7f495f85fb0e6e8f472ce7cf56ed1`
- 打包审计：`work/candidates-runtime-bound/component-v5-calibrated-test/package.audit.json`
- 真实生产适配器双加载 smoke：`artifacts/audits/component_v5_calibrated_candidate_runtime_provider_smoke_replay.json`

该候选的 bundle 会哈希绑定五类附件：验证校准、独立 test 确认、test 捕获总体、正式 parent status 和原始候选 bundle。若任一附件被修改，`build_candidate_lock_entry` 必须 fail closed。

### 4.2 Pin v5

- 运行时绑定候选：`work/candidates-runtime-bound/pin-v5-boundary/bundle.manifest.json`
- 当前 bundle SHA-256：`58f82ecabec7ad450003a030948e453bf0e6cedae708cbf62f435de2cd949e37`
- 已完成正式多指标门禁和生产运行 smoke；除非文件哈希失配，不要重训或回退到旧锁 Pin。

### 4.3 四模型确定性 Junction 架构

已完成：

- `production_provider` 按 pipeline 的 `learned_junctions_enabled` 解析精确任务集合；
- false 模式要求且只允许四个 learned task；
- `CompositePerceptionProvider.model_task_plan` 记录 required/loaded/invoked；
- 每个 IR 写入 model task plan、精确 model versions，且不得出现 quarantined Junction；
- CLI、HTTP、production evaluation、readiness 和 promotion 均使用同一任务计划；
- promotion 的 deterministic Junction 探针验证 T 形 mask 能产生一个 T junction 和三个端点。

不要恢复“恰好五模型”的旧假设。若未来重新启用 learned Junction，必须走单独的五模型模式和真实六类证据，不能与本轮混用。

## 5. 当前正在运行的训练编排

### 5.1 OCR v5

配置：`configs/train/dataset300_v2_ocr_v5_allgate.json`  
状态：`work/train-ocr-v5-allgate/torch_run_status.json`  
日志：`work/train-ocr-v5-allgate/training_log.jsonl`

四个硬门禁：

1. normalized exact match `>= 0.65`；
2. detector H-mean `>= 0.45`；
3. designator exact match `>= 0.90`；
4. pin-number exact match `>= 0.74`。

本文更新时已完成 epoch 5：

- normalized exact `0.684592`：通过；
- designator exact `0.912088`：通过；
- pin-number exact `0.755385`：通过；
- detector H-mean `0.408981`：未通过，是唯一剩余门禁；
- detector H-mean 从 epoch 1 的 `0.389757` 持续升至 epoch 5 的 `0.408981`。

训练最多 12 epoch，early-stop patience 为 6。Agent 必须等待完整终态：

- `candidate_evidence_ready`：进入 OCR 候选运行验证与 Wire 自动接力；
- `gate_failed`：不得放宽门禁，应先分析 detector precision/recall、阈值和训练曲线，再决定新配置；
- `running`：只允许监控，禁止启动第二个 OCR GPU 进程。

检查命令：

```powershell
Get-Content .\work\train-ocr-v5-allgate\training_log.jsonl -Tail 3
Get-Content .\work\train-ocr-v5-allgate\torch_run_status.json -Raw |
  ConvertFrom-Json | Select-Object status,registerable,finished_at_utc
```

### 5.2 Wire v5 自动接力

配置：`configs/train/dataset300_v2_wire_v5_fresh_allgate.json`  
日志：`work/wire-v5-launcher.log`

Wire 等待器只在 OCR 出现完整终态且四门全过时启动 fresh v5。不要终止等待器，也不要手动并发启动 Wire。若 OCR 失败，等待器应 fail closed；若 OCR 通过但 Wire 未启动，先检查日志和 watcher 进程，再修复启动契约。

Wire v5 必须补齐：

- fresh lineage，不复用缺失 lineage 的旧 formal4 作为最终证据；
- skeleton/centerline、connectivity、导出一致性与 bundle；
- production adapter smoke；
- 与 graph/net 拓扑的 E2E 结果，不得只凭 pixel F1 晋级。

## 6. 后续 Agent 的执行优先级

### P0-A：完成 OCR → Wire 训练链

1. 持续监控 OCR，每个新 epoch 记录四门状态，但不以中间轮次放行。
2. OCR 终态通过后，验证 artifact、predictions、model card、bundle、导出一致性和生产适配器实际加载。
3. 确认 watcher 只启动一个 Wire fresh v5 进程。
4. Wire 终态后做分层审计和生产 smoke。

完成定义：Component、Pin、OCR、Wire 四个任务各有一个未祝福但可复核、运行时参数哈希绑定的候选 bundle。

### P0-B：修复 release50 证据链四个绕过点（代码完成）

已完成，详见第 7 节。旧 summary、自选/partial manifest、错误 split、direct-y Net edge、伪造 view matrix 和 case manifest 篡改均已有 fail-closed 测试。注意：这只代表证据工具闭环，不能替代新锁下的真实 3×50 运行。

### P0-C：组装全新四模型锁

使用 `data_tools/model_promotion.py` 的 deterministic Junction 模式：

- 输入恰好四个候选 bundle；
- Component 使用本文第 4.1 节的新校准候选；
- Pin 使用第 4.2 节候选；
- OCR/Wire 只能使用本轮最终通过的候选；
- production config 必须保持 `learned_junctions_enabled=false`；
- 输出到新目录，不覆盖旧锁；
- promotion report 必须记录 deterministic Junction contract 与运行源码哈希。

完成定义：新锁中的四项均为 promotion 工具生成的 blessed/verified 条目，readiness 在相同 config 下严格返回 ready；旧 learned Junction 不在锁中。

### P0-D：三次 release50 严格端到端

使用同一四模型锁、同一 pipeline/profile、同一 50 例 manifest 连续执行三次：

- 三次均 50/50 成功；
- Validator 0 ERROR，Serializer 全部成功；
- 每次 summary 和 prediction manifest 都绑定运行时源码指纹；
- 每个案例的 official JSON 和 Circuit-IR 字节可追踪；
- 三次 `prediction_set_sha256`、逐案例 official SHA 和 IR 语义哈希满足确定性契约；
- 总时长均小于 2 小时，并报告 cold start、P50/P95、峰值显存；
- 任何 partial/not_run/error 都应使正式门禁失败。

### P0-E：坐标、trace、接口与离线联调

1. 从上述同一次正式 run 生成坐标证据，不允许重新包装旧 run。
2. 运行 release trace probe 和 determinism evidence packager。
3. CLI：单图、批量、错误输入、只读输入根、原子输出和 readiness。
4. HTTP：`/health`、`/ready`、`/v1/infer`，验证生产未就绪和推理失败的状态码。
5. 在可用的 Linux/GPU + Docker 主机上做离线 build、无网络启动和 50 例复演。

## 7. 发布证据链四项问题及当前实现状态

以下问题来自独立复核。代码级修复均已完成；在新锁产生正式 3×50 证据前，发布状态仍保持阻塞。

### 7.1 正式坐标报告未强制绑定 canonical release50

原问题：`generate_current_coordinate_evidence.py` 可以接受任意 workspace manifest；历史正例甚至用 1 row 也能 formal pass。

当前状态：已修复。正式重算只接受 canonical manifest、`release50`、精确 50 个唯一 case 和完整 duplicate-group membership；报告新增 `source_run_binding`，硬门禁要求 coordinate/performance/determinism 共享同一正式 run。

修复要求：

- 正式模式只接受相对路径 `data/splits/dataset300_release50_v1.jsonl`；
- summary.split 必须为 `release50`；
- manifest、prediction manifest、summary.coverage 和坐标 case manifest 均必须恰好 50 个唯一 case；
- 坐标报告必须与 performance/determinism 使用同一 `inference_run_id` 和 summary SHA；
- 1 例或其他 split 只能标记 diagnostic，release gate 必须拒绝。

### 7.2 旧 summary 可被当前源码重新包装

原问题：summary 没有记录推理当时的 runtime source manifest；generator 会把“今天的源码 SHA”附到旧 IR/summary 上。

当前状态：已修复。`infer_manifest` 在推理开始和 prediction manifest fsync 后各计算一次生产源码清单；summary 内嵌该清单。generator 会重算，release auditor 还会用独立实现再次枚举和哈希，历史 summary 在源码变化后不能重新包装。

修复要求：

- `pcbparse.evaluation.infer_manifest` 在真实推理开始和结束时计算 production runtime source manifest；
- summary 内嵌相对路径、字节数、SHA-256 和 source-set SHA；
- 运行期间源码变化必须 fail closed；
- generator 和 release audit 独立重算当前源码，必须与 summary 内嵌字节完全一致；
- 正式证据选择以原始 run completion/inference ID 为准，不以“重新包装报告时间”为准。

### 7.3 Net edge 的 H−y 检查与 serializer 共因

原问题：expected edge 使用 serializer 私有函数生成，之后又从 actual 反推 source；若 serializer 自身错误，两边可能一起通过。

当前状态：已修复。审计器直接从 Circuit-IR segment/pin 构造字面 `(x,H-y)` 期望，独立实现 pin-only zero-length 规则，不调用 serializer 的 geometry helper；direct-y 反例测试通过。

修复要求：

- 直接从 IR `segment.start/end` 和 pin point 读取 source `(x,y)`；
- 用审计脚本内的字面独立公式 `(x, image_height - y)` 构造 expected official endpoint；
- 每个 net 用 Counter 比较归属与重复数；
- 不调用 `_canonical_endpoints` 或 `_net_zero_length_geometries` 生成 expected；
- 增加“仅 edge serializer 错用 direct-y”负向测试，必须失败。

### 7.4 局部→全局 roundtrip 只是人工恒等式

原问题：`(local + offset) - offset` 只证明算术恒等式，没有比较模型 evidence 中的真实 local/heatmap 坐标与最终 IR global 坐标；`local_to_global_coordinate_evidence_checked=0` 仍可能放行。

当前状态：已修复。Component/OCR/Pin/Segment/Junction evidence 现在携带 `image_view_local_to_original_v1` 绑定；报告生成器独立重建每个 view，逐对象核对 local geometry、matrix、declared original geometry 和最终 IR，并要求五类对象在正式 50 例中均有正覆盖。view matrix 偏移篡改会被拒绝。

修复要求：

- Component：读取 local bbox 和 source view transform，与最终 global bbox 比较；
- Pin：读取 evidence 的 local/heatmap point、ComponentView origin/scale，与最终 pin/base 比较；
- Segment/Junction：读取 wire view local geometry与最终 global point/segment 比较；
- OCR：读取 local bbox、view ID 和 transform，与最终 text bbox 比较；
- 正式 50 例必须有非零、分模块的真实 evidence coverage；
- 任一 `_restore_pins`、`_restore_junctions` 或 bbox restore 注入偏移时，负向测试必须失败。

### 7.5 当前硬门禁诊断快照

最新诊断文件：`artifacts/audits/release_hard_gate_audit_post_contract_20260824.json`  
文件 SHA-256：`e60343d526948e8f4026c3007a4e2dd0ea290b7712cd714b2ce1ff3ea745f822`  
生成时间：2026-08-24 15:52:39（Asia/Shanghai）

该文件是**阻塞项盘点，不是正式发布证据**。当前结果：

- `locked_blessed_models`：`fail`，原因是主锁仍是旧的未祝福候选；只有 OCR/Wire 新候选完成后才能通过 promotion 工具生成新四模型锁；
- `official_json_zero_extra_fields`、`official_json_zero_dangling_references`、`coordinate_roundtrip_lte_0_5px`、`same_environment_three_run_json_hash`、`fifty_cases_under_two_hours_with_ten_percent_reserve`、`traceable_errors`：均为 `unverified`，需要新锁下的同源 canonical release50 三次运行；
- `offline_container`：`unverified`，需要在具备 Docker 的 Linux/GPU 目标机生成证据；
- `bless_authorized=false`、`promotion_authorized=false`、`release_ready=false`，这是当前正确且必须保持的 fail-closed 状态。

后续 Agent 不应通过修改审计 JSON 消除这些阻塞项。正确顺序是：完成 OCR/Wire 候选 → 工具生成并验证新四模型锁 → 同配置连续运行 3×50 → 从这些运行生成 coordinate/determinism/trace/performance 证据 → 目标机离线容器复演 → 重新运行硬门禁审计。

## 8. 建议的质量优化方向

在 P0 发布闭环完成后，再按以下顺序做模型和算法优化。

### 8.1 OCR detector 优先于 recognizer

- 当前唯一未通过门禁是 detector H-mean；不要继续只优化 exact match。
- 分析 detector FN 的尺寸、来源域、旋转、文本类型和 Component ROI/全局 tile 分层。
- 在固定 val 上比较 detector threshold，只能形成新的校准报告；独立 test 不做阈值搜索。
- 增加 designator/pin-number 的端到端 coverage：被检测、被识别、被关联、最终写入四个阶段分别计数。

### 8.2 Graph 拓扑优先于继续抬高 Wire pixel F1

- 做 oracle 拆解：GT Wire + predicted vectorizer、predicted Wire + GT junction、GT Pin + predicted graph。
- 报告 Net split、Net merge、pairwise connectivity、孤立段、短段、重复 junction，而不只报 pixel F1。
- 在冻结 val 上扫描 endpoint merge、pin snap、gap repair 和 crossing 参数；每轮固定参数后再用独立套件确认。

### 8.3 Component 长尾与域稳健性

- 当前 calibrated v5 已足够进入联调；先看 release50 的真实级联收益。
- 若仍重训，重点看 PIN/Net_short/Testpoint/Block 等弱类，不要只抬 Box/C/R/Gnd。
- 继续使用 512/64 Component tiling；任何 geometry/NMS 变化都需新的 val 选择和 test 确认。

### 8.4 Validator 错误预算

release50 失败时按以下优先级聚类：

1. Component missing/duplicate key；
2. Pin missing number/owner；
3. disconnected segments；
4. net split/merge；
5. serializer/schema；
6. timeout/OOM。

不得用 degraded serializer 隐藏 ERROR。正式通过仍要求 0 ERROR。

## 9. 测试与运行环境约束

- 当前 `.venv-train` 有 Python/Torch/CUDA，但没有 pytest；不要反复联网安装 pytest。
- 使用标准库 unittest：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\src').Path
& '.\.venv-train\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -q
```

- 当前工作树最新完整回归：503 项通过、3 项跳过、0 失败（2026-08-24）。阈值审计测试已移除不必要的 pytest 依赖并纳入标准 unittest；后续再改源码后仍须重跑，不能沿用该数字。
- 最近定向回归已通过：training foundation/candidate rebind/Component runtime smoke、四模型 readiness/provider/eval/CLI/API、release audit/coordinate/determinism/trace 的相关套件。
- 最新 release 定向结果：`tests.unit.test_current_coordinate_evidence` 10 项通过；runtime source 独立实现与生产实现对当前 91 个文件得到完全相同清单；Composite/OCR 坐标绑定相关 19 项通过。
- 当前主机未发现 Docker；不要伪造容器通过结论。
- 工作树可能包含用户和其他 Agent 的修改；只用 `apply_patch` 做局部编辑，不执行 reset/checkout/递归删除。

## 10. 后续 Agent 的最小检查清单

接手时：

- [ ] 阅读本文；
- [ ] 检查 active goal 与当前 plan；
- [ ] 查看 OCR status/log，确认没有重复 GPU trainer；
- [ ] 查看 `work/wire-v5-launcher.log`；
- [ ] 对 Component/Pin bundle 运行 `build_candidate_lock_entry`，确认附件和模型字节未漂移；
- [ ] 检查 `models/models.lock.json` 仍未被手工祝福；
- [ ] 读取 7.5 的硬门禁快照；若生产源码或证据文件已变化，重新运行审计并记录新文件哈希，不覆盖旧诊断；
- [ ] 运行与计划修改相关的最小 unittest。

交接前：

- [ ] 更新本文的时间、最新 epoch/终态和新 artifact SHA；
- [ ] 明确列出“已通过、未通过、正在运行”；
- [ ] 写明所有新进程、watcher 和输出目录；
- [ ] 不把 diagnostic 写成 formal；
- [ ] 不声称官方分数；
- [ ] 只有四模型锁、3×50、坐标/trace、接口和目标机演练全部完成后，才把目标标记 complete。

## 11. 最终完成定义

只有下列条件全部满足，系统才可以在本项目口径下称为“高度可用竞赛候选”：

- [ ] Component/Pin/OCR/Wire 四个 learned bundle 全部正式通过并运行时绑定；
- [ ] deterministic Junction 四模型 promotion 契约通过；
- [ ] 新 blessed lock 由工具生成且 readiness=ready；
- [ ] canonical release50 连续三次 50/50 成功；
- [ ] Validator 0 ERROR、Serializer 50/50 成功；
- [ ] 坐标 H−y、真实 local→global、trace、源码指纹和确定性证据全部通过；
- [ ] CLI/HTTP 真实模型接口联调通过；
- [ ] 目标 Linux/GPU 离线容器 50 例演练通过且总时长 <2h；
- [ ] 完整 unittest 回归无失败；
- [ ] 最终报告明确区分本地指标与官方未知得分。
