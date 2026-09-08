# 实施状态与验收证据

历史基线审计时间：2026-08-23 23:22（Asia/Shanghai）  
下方矩阵和指标段落是 2026-08-23 的历史基线快照；2026-08-27 的修复后状态见下方“修复后验收增量”，不要将历史候选状态与当前主锁混用。

## 2026-08-31 Wire 最小线段与 Pin 绑定正式口径复核

- 在当前正式模型锁、正式 Wire threshold `0.60` 和 canonical release50 上完成 Wire 最小线段长度的 image-only 复核。固定代表性4例中 `24px` 的加权代理最高（`0.467548`），但全量50例中 `16px/20px/24px` 分别为 `0.478267/0.474545/0.471474`，均低于当前正式基线 `0.481485`；因此不晋级，正式配置继续使用 provider 默认 `wire_minimum_run_px=4`。实验配置 `configs/pipeline.wire-minrun24-experiment.yaml` 仅保留为 opt-in 候选。
- 重新用正式模型锁和正式 Wire threshold 对 `association.pin_unique_margin=0.0` 做 release50 复核：`50/50` 成功，Component/Pin/Graph/Wire 分别为 `0.780685/0.516582/0.201286/0.476833`，与当前正式基线完全一致，Component 与 Pin 字段 exact 也没有变化；因此继续保持正式值 `0.02`。产物见 `work/release50-pinassoc-margin0-formalwire060/summary.json`。
- 当前图层仍以 `NET.DISCONNECTED_SEGMENTS` 和 Pin owner/字段绑定为主要阻塞；单纯过滤短 Wire 段或降低 Pin 文本唯一性门槛，尚未证明能够改善端到端结果。严格 Validator、官方 Serializer、官方 evaluator、目标 GPU P95 和三次确定性 hash 仍未完成，不能宣称达到赛会95%。
- 对现有50份 IR 反事实开启 POWER/GROUND 标签合并，及将 SIGNAL 标签最低分/合并阈值放宽至 `0.50/0.60`，Graph F1 均保持 `0.201286`，说明当前没有足够的跨物理网高置信标签证据；正式配置不放宽标签门槛。
- 未晋级 Pin 权重 `models/candidates/pin/model.onnx` 在正式 `0.75/max12/margin16` 四例探针上将 Pin F1 `0.517361→0.470187`、Graph F1 `0.104430→0.094571`，已拒绝，不进入模型锁。
## 2026-08-31 Pin 阈值0.75与Wire过滤全量复核

- 在当前正式模型锁和 canonical `release50` 清单上完成 Pin score `0.75/max12` 的 `50/50` image-only 生产复核，耗时 `867.719s`（约 `14.46min`），无失败样本。相对当前 `0.70/max12`：Component F1 保持 `0.780685`，Pin F1 `0.439441→0.441727`，Graph pairwise F1 `0.116905→0.116502`，Wire F1 保持 `0.476833`；按赛会权重的 raw-v1 加权代理 `0.432666→0.433096`，因此已将发布候选 `configs/pipeline.release-candidate.yaml` 的 Pin 默认阈值由 `0.70` 调整为 `0.75`。
- 同口径 Pin score `0.80/max12` 全量结果为 Pin F1 `0.442907`、Graph F1 `0.113946`、加权代理约 `0.432497`，因拓扑下降而拒绝；`0.75` 是当前已完成全量复核的折中点，不代表官方成绩。
- Wire `threshold=0.70` 在4例探针上将 Wire F1 降至 `0.478181`，拒绝；`wire_minimum_run=6` 在4例上曾短暂改善Graph，但完整50例变为 Wire F1 `0.451402`、Graph F1 `0.118266`，低于当前 Wire `0.476833`，拒绝，正式配置继续保持最小线段 `4px`。
- 更新配置后的 `0024-other.png` 严格生产推理仍以 `29` 个 ERROR fail-closed，入口返回非零，没有生成可提交输出；`healthcheck --require-ready` 返回 ready，模型锁未修改。默认回归为 `476` 项通过、`67` 项环境跳过，集成测试 `48` 项通过。
- 最新主阻塞仍是 Graph/NetHypergraph 低召回、Component/Pin 上游召回与语义绑定不足；当前离官方95%目标仍有显著差距，下一阶段应优先提升模型/域专门化和Pin owner质量，而不是继续扩大几何容差。
## 2026-08-31 Pin 框边界 margin16 全量提升

- 在不更换 Pin 模型、不改变 `models/models.lock.json` 的前提下，将 `pin.box_margin_px` 从 `4.0` 调整为 `16.0`，并完成 canonical release50 `50/50` image-only 评测；产物见 `work/release50-pin-margin16-canonical/summary.json` 与 `reports.json`。
- 相比当前 Pin `0.75/max12/margin4`：Component F1 保持 `0.780685`，Pin F1 `0.441727→0.516582`，Graph pairwise F1 `0.116502→0.201286`，Wire F1 保持 `0.476833`；按赛会权重的 raw-v1 加权代理 `0.433096→0.481485`。Pin precision/recall 为 `0.553587/0.484215`，Graph precision/recall 为 `0.298627/0.151804`，说明主要收益来自 Pin 召回提升而非单纯放宽误检。
- 域分层微平均方向一致：AD Pin/Graph 约 `0.43/0.11→0.49/0.16`，Datasheet `0.37/0.10→0.42/0.16`，JLC `0.55/0.17→0.65/0.32`，KiCad `0.09/0.01→0.17/0.03`；不是单一域样本驱动。
- `0024-other.png` 在 margin16 单图严格 Validator 下仍 fail-closed，错误为 `COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=10`；相对当时 margin4 基线没有新增正式 ERROR，因此先将 `configs/pipeline.release-candidate.yaml` 的 Pin margin 提升到 `16.0`，但仍未宣称发布就绪。
- 当前机使用 CPU fallback（ONNX CUDA 缺少 `cublasLt64_13.dll`），50 例耗时 `871.813s`，在本机 7200s 预算内；目标 GPU P95、10% reserve、三次严格成功复跑和官方 evaluator 仍待完成。
- 随后以已晋级的 margin16 正式配置执行严格 release50：`50/50` 均被 Validator fail-closed，耗时 `891.390s`，P95 单例 `53.035s`，虽满足本机时间窗但没有生成可提交 official JSON。`COMPONENT.MISSING_KEY=1,784` 保持不变，`NET.DISCONNECTED_SEGMENTS=1,681`（较前次 `+1`），说明 Pin 召回收益尚未转化为 Component key 召回与网络闭合收益；产物见 `work/production-e2e-pinmargin16-candidate/summary.json`。
- 在固定 margin16 IR 上扫描 Graph 端点、吸附、RDP、共线和 junction 变体，当前配置的 `unique_margin=0.5` 仍为最优；没有证据支持继续放宽 Graph 参数，扫描记录见 `work/graph-variant-grid-pinmargin16.json`。
- 在 margin16 基础上将 `association.component_unique_margin` 从 `0.02` 降为 `0.0`：50 例 Component/Pin/Graph/Wire 及 raw-v1 加权代理保持不变（`0.481485`），但 Component designator `14.46%→17.54%`、value `10.84%→16.16%`；严格 canonical release50 的 `COMPONENT.MISSING_KEY` 进一步降至 `1,699`（较 margin16 减少 `85`），`NET.DISCONNECTED_SEGMENTS=1,681` 不变，没有新增 ERROR 类别。最终严格产物见 `work/production-e2e-componentmargin0-pinmargin16/summary.json`，配置哈希为 `74570c29...`。
- 已刷新 `work/audit-release-hard-gates-componentmargin0-pinmargin16.json`：`release_ready=false`。当前缺口包括 official JSON 零多余字段/零悬挂引用、坐标往返、三次确定性、离线容器、锁定模型与可追踪错误证据；这些不是通过再次重复同一条失败链路可以补齐的。
## 2026-08-31 正式入口当前锁绑定基线

- 使用当前 `models/models.lock.json`、`configs/pipeline.release-candidate.yaml`、`competition` profile 和 canonical `release50` 清单完成一次正式 image-only 入口复演；summary 已绑定当前配置哈希 `e102b696...`、模型锁哈希 `af35e129...`、运行 ID、源码清单和预测 manifest。
- 运行覆盖 `50/50`，总耗时 `896.828s`（约 `14.95min`，在 7200s 总时限内），但严格 Validator 拒绝全部 `50/50`，因此没有任何可提交 official JSON。主要诊断累计为 `COMPONENT.MISSING_KEY=1,784`、`NET.DISCONNECTED_SEGMENTS=1,680`、`ASSOCIATION.PIN_TARGET_AMBIGUOUS=15,447`、`ASSOCIATION.PIN_TARGET_UNKNOWN=9,222`；这证明当前硬阻塞是上游 Component/Pin owner 召回与网络拓扑闭合，而不是单纯运行速度。
- 本次正式失败产物见 `work/release50-formal-pin075-run1/summary.json` 与 `predictions.jsonl`。发布硬门禁报告 `work/audit-release-hard-gates-pin075.json` 仍判定 `release_ready=false`：锁内四个已用模型的文件哈希一致，但 Junction 免除项缺少当前正式运行证据；官方 JSON、坐标往返、三次确定性、离线容器和可追踪错误证据仍未验证。
- 结论：不继续重复无法通过 Validator 的 50 例运行；下一条高收益路径必须降低 `COMPONENT.MISSING_KEY` / Pin 目标歧义并提升有效 Pin owner，再重新做严格 release50。未获得官方 evaluator 前，不把本地 raw-v1 代理分数当作赛会95%成绩。
## 2026-08-31 OCR 候选识别固定难例复核（未晋级）

- 新增默认关闭的 CTC prefix-beam 候选识别通道，并在语义层增加受限候选回退；候选只允许补救原始 OCR 未分类文本，不覆盖已有字段，也不改变当前正式模型锁。
- 在固定高损失样本 0087、0029、0235、0096 上做 image-only 对照：候选数 `0` 与候选数 `3` 的 Component、Pin、Graph、Wire 四项端到端指标完全相同；0029 虽有 `695/695` 个 token 生成备选、`79` 个语义字段被候选暂时救回，但没有形成可评分的最终字段增益。
- 候选解码 4 例耗时 `60.657s`，关闭候选为 `56.516s`，额外开销约 `7.3%`；以当前无收益证据，不运行全量50例、不进入正式配置。
- 新增确定性 CTC 回归用例后，完整回归为单元 `477` 项通过、`68` 项环境跳过，集成 `48` 项通过，0 失败；`healthcheck --require-ready` 仍为 ready，四个锁定模型哈希未改变。当前主机仍因 ONNX CUDA 依赖缺少 `cublasLt64_13.dll` 使用 CPU fallback，目标 GPU 性能与离线容器仍未验收。
- 结论：OCR 候选不是当前可证明的高收益路径；下一优先级仍是 Component/Pin owner 召回、Pin number/name 语义绑定和 Graph/NetHypergraph 拓扑闭合。
## 2026-08-31 OCR 检测最小面积校准复核（未晋级）

- 为验证 OCR v8 验证集的 detector Hmean 校准信号，新增并接通 `detector_min_area` 的生产运行时覆盖、配置解析和探针参数；参数只作为实验覆盖，不改变 `models/models.lock.json` 和正式 `pipeline.release-candidate.yaml`。
- 在 OCR v8、detector threshold `0.2` 下，验证集扫描显示 `min_area=5..8` 可使 Hmean 从原门限下的 `0.4489` 提升至 `0.4501..0.4528`，但这只是 OCR 检测器验证指标，尚未证明端到端收益。
- 固定高损失样例 `0087/0029/0235/0096` 做 image-only 对照：`min_area=6` 和 `min_area=8` 的 Component F1 均为 `0.533333`、Pin F1 均为 `0.076825`、Graph F1 均为 `0`、Wire F1 均为 `0.515981`，与基线完全一致；两次运行均 `4/4` 成功、无错误，单次约 `64.6..65.0s`。
- 结论：验证集 Hmean 的小幅越门不能转化为当前 E2E 组件/Pin/拓扑收益，面积阈值候选不扩大至 release50，不进入正式配置。运行时参数化能力保留，后续若有新 OCR bundle 可直接复用。
- 本轮正式运行时全套回归 `525` 项通过、`68` 项环境跳过；训练环境另有 `2` 个既有冻结物化文件错误（提交标记/Windows 权限），与本轮源码无关。`healthcheck --require-ready` 仍为 `ready`，模型锁哈希未改变。
## 2026-08-30 赛会宣讲视频要求对齐增量

- 已将视频中的发布约束落到回归框架：单图最长 600 秒、全批次最长 7200 秒，并在每个成功/失败样本中记录实际模型、配置哈希、阶段耗时、剩余时间和 Token 使用；超时样本明确标记为 `not_run`，不伪造结果。
- 组件 Key 全局解析已改为保守策略：规范化 OCR 文本、显式组件归属硬约束、组件框间距评分、重复 Key 全局冲突拒绝。默认关闭 Pin OCR 候选硬门控，因为 0024 端到端证据显示它没有降低严格错误且会丢失 Pin。
- 组件 512/64 小切片实验配置已补齐 2 小时总时限、候选阈值 `0.2` 和候选上限 `128`。冻结验证集组件 F1 约 `0.753`、Precision 约 `0.741`、Recall 约 `0.764`，但 0024 严格端到端错误为 `22`，高于默认 1024/160 的 `12`，因此仍是候选配置，不得升为 baseline。
- 本轮验证：519 项测试中 514 项通过、3 项跳过、2 项错误；错误仍为冻结训练物化文件的提交标记/Windows 权限问题。默认 0024 仍未达到严格零错误，当前不能声称官方 95% 或竞赛发布就绪。
- 已修复生产适配器的 `device=auto` 判定：现在同时检查 CUDA 可用性和实际设备数；在 `CUDA_VISIBLE_DEVICES` 为空的 CPU 环境下，主锁可正常加载，不会误走 `cuda:0`。
- 已加入严格的 pipeline 级 Pin/Wire 推理覆盖，覆盖参数进入 perception_config 审计，不改变 models/models.lock.json 的 artifact 身份。Pin 候选继续使用 filled_rect_v1 / score_threshold=0.5 / max_detections_per_component=4 / box_margin_px=4.0；Wire 阈值覆盖仅作为独立实验开放。
- 主模型 raw-v1 三例 A/B（0024、0017、0029）显示该 Pin 候选 F1 分别约 `0→0.390`、`0.056→0.187`、`0→0.106`，Wire F1 不变；因此候选方向保留，但在完成 30–50 例门禁前不替换默认配置。
- Key 关联已增加显式 ROI 归属优先规则；在 0024 上实际非空 Key 仍为 `2/14`，说明当前主要阻塞是 OCR 设计符号召回与多候选冲突，不是单纯的字段选择顺序。线段距离过滤在不同样例上收益不稳定，暂不默认启用。

## 2026-08-30 组合候选 release50 验证

- 已将组件小切片与 Pin 候选组合为 configs/pipeline.component512-pinfilled-experiment.yaml：Component 使用 512/64 tile、候选阈值 0.2；Pin 使用 filled_rect_v1、阈值 0.5、单组件最多 4 个候选；物理图使用 0.75px 内部重叠和 6px 保守共线间隙合并，但默认主配置仍关闭。
- 在同一主模型锁下按 release50 清单完成 50/50 图像推理与 raw-v1 局部评分；产物见 work/release50-component512-pinfilled/summary.json 和 reports.json。总耗时约 845.6s（约 14.1 分钟），但该次跳过严格 Validator/官方 Serializer，因此只能作为候选性能与耗时证据。
- raw-v1 聚合：Component F1 0.7503（P/R 0.7403/0.7606），Pin F1 0.3781（P/R 0.4915/0.3072），Wire F1 0.4479（P/R 0.2917/0.9639），Graph pairwise F1 0.0954（P/R 0.2583/0.0585）。三张重点样例 0024/0017/0029 的 Component F1 为 0.745/0.702/0.877，Pin F1 为 0.667/0.389/0.222。
- 域分层显示 KiCad 是明确异常域：Component/Pin/Graph 平均 F1 约 0.257/0.065/0.000；AD 为 0.755/0.294/0.048，Datasheet 为 0.767/0.390/0.141，JLC 为 0.802/0.469/0.109。后续应先做域识别与尺度分流，不宜继续用全局阈值硬调。
- 组件字段仍是语义瓶颈：50-case 端到端有效字段中，designator/value/searchName 的 normalized exact 约 5.38%/1.52%/0%；Pin number/name 约 1.99%/0.49%。Graph 召回仅 5.26%，说明主要损失在 Pin owner、Pin number 和 Pin—Wire 网络归属，而不是 Wire 像素召回。
- 共线合并单元测试已覆盖；在同一 50-case IR 上扫描间隙容差 2/3/4/6/8px 后，6px 为当前最佳 Graph F1 0.0954，8px 精度明显下降，因此主组合保留 6px、默认主配置仍关闭。
- 组合候选的 production healthcheck（含真实模型加载、Pin 覆盖参数审计）通过；模型锁未改。组合候选对 0024 运行严格 Validator/Serializer 时仍有 22 个 COMPONENT.MISSING_KEY，官方 evaluator 也缺失，因此候选状态保持 keep_opt_in，不能宣称官方 95% 或可直接提交。
- 最新全量测试：519 项，514 项通过、3 项跳过、2 项错误；两个错误仍是冻结训练物化文件的既有提交标记/Windows 权限问题。

## 2026-08-30 Wire 阈值与性能分流实验

- 新增 production Wire threshold override，模型锁仍保持原 artifact 和 0.5 默认值；配置与 trace 都能审计实际生效阈值。
- Wire 0.55 的 50-case raw-v1 结果为 Wire F1 0.4677（P/R 0.3102/0.9508），Graph F1 0.0879；Wire 0.60 为 Wire F1 0.4860（P/R 0.3280/0.9372），Graph F1 0.0787。两者 Wire 精度提升，但 Graph recall 下降，均不替换当前组合主候选。
- 0024/0017/0029 的 Wire512/64 瓦片实验没有超过 Wire1024/160；0011 的 Wire overlap256 也没有改善 Wire/Graph，且线段数和耗时增加，全部拒绝。
- Pin max16 隔离实验在 0024 由 68 增至 73 个候选，但 Pin F1 从 0.6667 降至 0.6429，说明当前瓶颈是模型候选质量而非简单上限，主候选保持 max4。
- 当前未加入 Wire 后处理时，全局综合最优仍为 component512-pinfilled + gap6，50-case 四项加权 raw-v1 约 0.3978；该数值仅用于候选排序，不是官方成绩。
- 新增 Wire 1px 闭运算候选：在同一 OCR v8 + Pin 宽松关联组合上完成 50/50 image-only raw-v1 评测，产物为 work/release50-ocr8-pinassoc-loose-wire-close1/summary.json。Component/Pin 不变，Wire F1 0.4479→0.4408，Graph F1 0.0954→0.1086（P/R 0.2383/0.0703），四项加权 0.3978→0.4017，总耗时约 806.2s，无失败样本。该候选改善了网络召回但增加误连，暂设为 opt-in，不修改正式默认配置。
- 新增可复现候选配置 configs/pipeline.ocr8-pinassoc-loose-wireclose1-experiment.yaml，并将 graph.wire_gap_closing_radius_px 接入生产配置解析、校验和 perception trace；OCR v8 artifact SHA 为 50b9514f3b4c448b194d93cd278c0af106e95c007e4595442f225e12413fe266，仍因验证 Hmean 0.4489<0.45 且 bundle 不具备可登记正式证据而不得写入 models.lock.json。
- OCR v8 的验证集 detector threshold 扫描选中 0.2，Hmean 0.445864→0.448875，仍未跨过 0.45；三例和 50 例端到端均未带来对象/图收益，正式候选继续使用阈值 0.5。历史 Wire 模型和 RDP 激进简化也均已拒绝。
- 可复核产物：work/release50-component512-pinfilled/final-summary.json、work/release50-component512-pinfilled/gap-scan-summary.json、work/release50-component512-pinfilled-wire055/summary.json、work/release50-component512-pinfilled-wire060/summary.json、work/ocr-v8-threshold-selection.json。

## 2026-08-30 上一阶段联合候选（Pin score 0.90）

- 可复现配置为 `configs/pipeline.component030-nms045-pinmax8-score09-ocr8-wireclose1-semantic-boxsearchname-experiment.yaml`：Component 候选门限 `0.30`、NMS `0.45`、512/64 tile；Pin 使用 `filled_rect_v1`、score `0.90`、单组件最多 `8` 个；Wire 使用 1px 闭运算；Graph 使用端点 `1px`、共线间隙 `6px`、Pin 吸附上限 `10px`；语义关联使用 Component 最低分 `0.30`、唯一 margin `0.02`、距离因子 `3.0`。
- 在同一正式模型锁下完成 50/50 image-only raw-v1 评测，全部成功、无失败，耗时 `799.5s`（约 13.3 分钟）。Component F1 `0.7723`（P/R `0.7987/0.7476`），Pin F1 `0.4183`（`0.6190/0.3159`），Wire F1 `0.4408`（`0.2853/0.9689`），Graph pairwise F1 `0.1030`（`0.2150/0.0677`），按赛会权重的本地加权值 `0.4164`。
- 相比前一组保守 Pin/Graph 全量候选（加权约 `0.4113`、Graph F1 `0.1114`），本候选 Pin F1 提升但 Graph F1 下降；因此推荐作为“加权平衡候选”保留，不覆盖正式保守 baseline。若优先拓扑，继续使用 Graph F1 `0.1114` 的 max4 候选。
- 语义字段全量结果为 Component designator/value/searchName `14.00%/10.84%/0%`，Pin number/name `4.64%/0.56%`。Box 专属 searchName 规则已经能产生候选，但代表性样本中出现 `Heade 20` 等 OCR 误识别，尚未带来 exact 命中；禁止用固定词典、自动编号或重复 Key 填充。
- 域均值仍显示明显域偏差：AD `Component/Pin/Graph/Wire=0.7606/0.3697/0.0430/0.3613`，Datasheet `0.7927/0.4246/0.1403/0.5386`，JLC `0.8295/0.5196/0.1343/0.4906`，KiCad `0.2399/0.0567/0/0.5337`。KiCad 是后续最高优先级专门化对象。
- 生产配置解析、真实模型锁 readiness 和 38 项受影响单测均通过；全套回归为 519 项，514 通过、3 跳过、2 错误，错误仍来自冻结训练物化提交标记/Windows 权限。该候选仍未通过严格 Validator/官方 Serializer 三次复跑，也没有官方 evaluator，因此不能声称达到官方 95%。
## 2026-08-30 Pin 阈值全量复跑

- 在同一主模型锁下，对离线高召回 Pin 缓存筛选出的五组组合进行全量比较；`score=0.70/max=8` 的 raw-v1 加权代理分最高，为 `0.420449`，高于 `score=0.80/max=8` 的 `0.419889`、自适应大框候选的 `0.419510` 和原 `score=0.90/max=8` 的 `0.416837`。
- 新增可复现配置 `configs/pipeline.component030-nms045-pinmax8-score07-ocr8-wireclose1-semantic-boxsearchname-experiment.yaml`。它保留 Component `0.30/NMS 0.45`、Pin ROI `64px`、filled_rect_v1、单组件最多 `8` 个、Wire 1px 闭运算、Graph 共线间隙 `6px` 等已验证设置，仅将 Pin score threshold 调整为 `0.70`。
- 真实生产链路 50/50 复跑产物为 `work/release50-pin-roi64-score07-max8-source/summary.json`，无失败样本；Component F1 `0.7721`、Pin F1 `0.4196`、Wire F1 `0.4408`、Graph pairwise F1 `0.1138`，加权代理分 `0.420449`。相对上一组 `0.4164` 候选提升约 `0.0040`，且 Graph F1 提升约 `0.0108`。
- 该次复跑耗时 `1055.8s`，因本机缺少 `cublasLt64_13.dll` 而使用 ONNX CPU fallback，不能替代目标 GPU 的 P95 证据；models/models.lock.json 未修改，仍未完成官方 evaluator、严格 Validator/Serializer 与三次确定性复跑，因此不能声称官方 95%。
- 对 `0024` 追加严格 Validator/Serializer 单图验证，CLI 返回非零并正确阻断输出：共 `20` 个 ERROR，其中 `COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=1`；因此 score=0.70 只改善候选排序，尚未解决发布级语义键和拓扑阻塞。

## 2026-08-30 Graph 容差全量筛选

- 在固定的 50 个 score=0.70/max=8/ROI64 生产 IR 上，仅重建 Physical/Electrical Graph，比较端点容差、共线合并、间隙修复、Pin 吸附和确定性 junction union；结果写入 `work/graph-variant-grid-score07.json`，没有重新使用标注或改变模型输出。
- `endpoint_merge_tolerance_px=1.5` 为当前最优：Graph F1 `0.116410`、P/R `0.2250/0.0783`，加权代理分 `0.421353`；基线 `1.0px` 为 Graph F1 `0.113825`、加权代理分 `0.420449`。端点 `2.0px` 略低，关闭 junction union 降至 `0.043931`，因此保留 junction union。
- 已新增可复现配置 `configs/pipeline.component030-nms045-pinmax8-score07-endpoint15-ocr8-wireclose1-semantic-boxsearchname-experiment.yaml`。该配置已通过 production readiness；对 `0024` 的严格 Validator/Serializer 仍正确 fail-closed，ERROR 组成不变为 `COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=1`。
- 该 Graph 结果属于固定生产 IR 的确定性后处理证据，尚未重新生成 50 次模型输出；全链路耗时沿用 score=0.70 复跑的 `1055.8s` CPU fallback，目标 GPU P95 仍待验证。

## 2026-08-30 语义规则与 Graph 组合增量

- 修复 Box `searchname` 分类规则中的正则转义错误：原实现把空白/分隔符写成了字面量反斜杠，短标题（例如 `USB C`）无法进入搜索名分类；修复后保留长标题行为，并新增回归测试。该修复只影响未带权威 `details` 的 OCR token，不注入标注内容。
- 在固定 `work/release50-pin-roi64-score07-max8-source/ir` 的 50 个生产 IR 上做 Graph 离线反事实筛选：`endpoint=1.5px + unique_margin=0.5` 的 Graph F1 为 `0.120963`（P/R `0.226318/0.082539`），加权代理 `0.422947`；当前 score07 基线为 Graph F1 `0.113825`、加权代理 `0.420449`。该结果只重建 Physical/Electrical Graph，不重新运行模型。
- 新增实验配置 `configs/pipeline.component030-nms045-pinmax8-score07-endpoint15-margin05-ocr8-wireclose1-semantic-boxsearchname-experiment.yaml`，已通过 `healthcheck --require-ready`。对 0024 的严格推理仍 fail-closed，`COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=1` 的 ERROR 组成未改变；因此该配置只能作为 opt-in 候选，不覆盖模型锁或正式 baseline。
- 新增 Graph 筛选脚本的 `--only` 选择能力和 unique-margin 候选记录；完整筛选产物为 `work/graph-variant-targeted-score07.json`。目标机 GPU 的 P95 和官方 evaluator 仍未验证。
- 完整回归：当前 `524` 项测试全部通过，`67` 项按环境条件跳过；本轮未引入测试回归。
## 2026-08-30 当前阻塞排序与优化方法

1. **P0：Graph/NetHypergraph 召回与误连。** 当前发布候选 `work/release50-adaptive015-graphmargin05-source` 的 Graph F1 为 `0.116905`，加权代理分为 `0.432666`；相较自适应基线 Graph `0.113014`、加权 `0.431304` 有小幅提升。主要损失仍来自 Pin—Wire 归属和大连通块误连；继续保持 learned junction 与 virtual gap repair 关闭。
2. **P0：组件与 Pin 语义字段。** 最新 Component detection F1 `0.780685`、Pin detection F1 `0.439441`；端到端字段 exact 为 designator/value/searchName `14.46%/10.84%/0%`，Pin number/name `4.77%/0.63%`。重复前缀规则已在完整 50 例重跑中将 designator exact 从 `90` 提升至 `94`，但主瓶颈仍是 OCR 设计符号召回、Pin number 识别与 owner 绑定；禁止自动编号或伪造字段绕过门禁。
3. **P1：KiCad 域专门化。** 当前发布候选 KiCad 域 Component/Pin/Graph/Wire 为 `0.4131/0.0870/0.0087/0.5883`，仍显著低于其他域；应建立版式路由和独立域验证，避免全局门限继续伤害其他域。
4. **P1：严格 E2E 发布门禁。** 当前全量 probe 只覆盖 perception、association、physical/electrical graph 与 raw-v1 评估，尚未完成零 ERROR 的 Validator/官方 Serializer、三次同环境 JSON hash 和官方 evaluator 对齐；模型锁虽 ready，但发布仍应 fail-closed。
5. **P2：性能与可复现性。** 最新发布候选 50 例为 `861.641s`（约 `14.36min`，本机 CPU fallback），仍低于 2 小时窗口，但不能外推目标 GPU P95；尚未完成目标 Linux/GPU 镜像、显存峰值、10% reserve 和三次重复 hash 证据。本机 ONNX CUDA provider 因缺少 `cublasLt64_13.dll` 回退，当前时延只作保守参考。
## 2026-08-30 自适应组件回退严格全量验证

- 自适应组件回退已增加三重 image-only 门控：主组件数不超过 `18`、`Box` 占比为 `0`、主组件框总覆盖率不超过图像面积 `2%`；不读取 `source_domain`，并将主结果、触发状态、回退候选数和参数写入 `component_candidate_gate`。
- 严格候选配置 `configs/pipeline.component030-nms045-pinmax8-score07-adaptive256-ocr8-wireclose1-semantic-boxsearchname-experiment.yaml` 在 release50 的 `50/50` 案例完成，`0` 失败、`0` 错误，耗时 `1382.7s`；仅 `0091` 触发回退。
- raw-v1 本地 proxy 微平均：Component F1 `0.7790`、Pin F1 `0.4224`、Graph F1 `0.1143`、Wire F1 `0.4408`，按赛会权重计算的加权代理 `0.423395`，较原 score07 基线 `0.420449` 提升 `0.002947`。KiCad 域 Component/Pin/Graph 分别由 `0.2347/0.0324/0` 提升至 `0.4131/0.0870/0.0087`，AD、Datasheet、JLC 未变化。
- 该提升只证明回退在当前 release50 上是低风险的小幅增益，不能解释为官方 95% 分数；官方 evaluator、目标 GPU P95/显存和三次重复 hash 仍未完成。当前运行因本机缺少 `cublasLt64_13.dll` 使用 CPU fallback，`1382.7s` 不能代表目标 GPU 时延。
- 在相同严格回退配置上将 Wire mask threshold 从 `0.50` 提升至 `0.60` 后，release50 仍为 `50/50`、`0` 错误；Wire F1 `0.4768`，Graph F1 `0.1107`，最终加权代理 `0.425730`，较严格回退候选 `0.423395` 和原基线 `0.420449` 分别提升 `0.002335` 和 `0.005281`。最终候选配置见 `configs/pipeline.release-candidate.yaml`，但 Graph 的小幅回落必须在官方 evaluator 上复核。
- 对 `0024-other.png` 做最终候选的严格 Validator/Serializer 检查仍 fail-closed：`29` 个 ERROR，其中 `COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=10`，没有生成可提交 JSON；这仍是当前发布阻塞，不允许通过自动补键或自动编号绕过。

## 2026-08-30 自适应回退与 Graph margin 发布候选复核

- 当前发布配置为 `configs/pipeline.release-candidate.yaml`，正式模型锁仍为 `models/models.lock.json`；在 canonical `release50` 清单上完成 image-only 生产链路 `50/50` 成功、`0` 失败，最新产物见 `work/release50-adaptive015-graphmargin05-semantic-doubled-prefix-source/summary.json` 与 `reports.json`。
- 最新 raw-v1 本地代理：Component F1 `0.780685`、Pin F1 `0.439441`、Graph pairwise F1 `0.116905`、Wire F1 `0.476833`，按赛会权重的加权代理为 `0.432666`；语义规则复跑总耗时 `861.641s`。该代理不是官方成绩，距离目标 95% 仍有很大差距。
- 已将自适应组件回退门控收紧为主组件数 `24`、组件框占图比例 `0.15`、总框面积占图比例 `0.02`，并将 `association.unique_margin=0.5` 纳入发布候选；两项变化在固定 IR 离线筛选与全链路重跑中方向一致，但收益仍属于小幅增益。
- 对 `0024-other.png` 的严格 Validator/Serializer 仍 fail-closed：`29` 个 ERROR，其中 `COMPONENT.MISSING_KEY=19`、`NET.DISCONNECTED_SEGMENTS=10`，没有生成可提交 JSON；官方 evaluator、目标 GPU 性能门禁和三次确定性 hash 仍缺失。
- 已加入 C/R 重复前缀 OCR 保守归一化：仅当组件类别与 OCR 前缀一致时将 `CC13`/`RR8` 形式折叠为 `C13`/`R8`，保留原始 OCR 文本和证据。对既有 50 例 IR 的离线反事实分析修复 `4` 个已匹配 designator 错误，未产生 Key 冲突；新回归测试通过。
- OCR 冲突文本合并仍为显式 opt-in，三例实验未改变指标，因此正式候选保持关闭；以 `.venv-train` 运行的 readiness healthcheck 为 `ready`，但本机 CUDA provider 因缺少 `cublasLt64_13.dll` 回退 CPU。
- Pin score `0.50/max12` 的 4 例对照虽提高召回，但 Pin F1 `0.5280→0.5184`、加权代理 `0.46338→0.46305`，并增加误连，故正式候选继续使用 `0.70/max12`。
- 固定最新 IR 的 Pin-snap 扫描结果：`4px` Graph F1 `0.103123`、`8px` `0.113600`、`12px` `0.116813`，均不超过当前 `6px` 的 `0.116905`；没有修改拓扑参数。
- 当前全套回归为 `524` 项通过、`67` 项环境跳过、`0` 项失败；新语义规则已完成 50 例端到端复跑，组件/Pin/Graph/Wire 几何指标保持不变，designator exact 提升 `4` 个。

## 2026-08-31 Component v6 低学习率微调复核与召回上限

- 新增候选配置 `configs/train/dataset300_v2_component_v6_finetune.json`：从 Component v5 权重严格 warm-start，学习率 `1e-5`、768×768、4 个 epoch 无改善即 early-stop；preflight 通过，300 张样本、600 个源文件和五任务 `147,758` 条记录全部校验通过，未改变正式模型锁。
- CUDA FP16 烟测在首个 batch 的损失全部为非有限值；切换为显式 FP32 后训练、验证和 TorchScript 导出烟测通过，确认是本机 torchvision/CUDA 组合的 FP16 数值稳定性风险，而不是数据血缘问题。
- 正式 FP32 微调在 epoch 1/2/3/4 的 `bbox_map50` 分别为 `0.237579/0.232161/0.230990/0.212018`，最佳值仍低于 `0.25` 门禁；候选正确 fail-closed，产物见 `work/train-component-v6-finetune/`，不得替换当前 Component 模型。
- 最新 release50 生产 IR 的硬上限分析：合格匹配 Pin 为 `1,854/5,163=35.91%`；即使这些 Pin 的网络连接全部重建正确，Graph pairwise 召回理论上限也只有 `970/5,876=16.51%`。已匹配组件所覆盖的 GT Pin 为 `3,889/5,163=75.32%`，说明 Component 召回和 Component→Pin owner 绑定是 Graph 之前的上游上限。
- 因此优化顺序调整为：先做 Component/Pin 联合召回与 KiCad 版式专门化，再做 Pin number/owner 和 Pin—Wire 归属；Graph 容差只做误连抑制，不能单独把系统推到 95%。继续保持严格 Validator/Serializer fail-closed，不允许自动编号或伪造字段填补缺失键。
- OCR v8 训练产物虽改善验证识别，但 detector Hmean 仍低于门禁，且 50 例 image-only swap 未改善端到端几何指标；继续保持未登记状态，正式链路只使用模型锁中的已审核 bundle。
## 2026-08-31 Pin ROI 与 Component 回退门控反例

- 固定 `0017/0024/0077/0166` 做 Pin ROI padding 对照：`32px` 的 Pin F1 为 `0.323140`、Graph F1 为 `0.099931`；`96px` 的 Pin F1 为 `0.449761`、Graph F1 为 `0.071545`，均低于当前 `64px` 配置在同一组样例上的 Pin F1 约 `0.527969`、Graph F1 约 `0.117647`，因此保留 `64px`。
- 固定 `0087/0092/0011/0017` 放宽 Component 回退门控至 `64` 个主框、框占比 `0.20`、总面积占比 `0.08` 后，Component/Pin/Graph/Wire F1 为 `0.329571/0.184758/0.013994/0.490567`，加权代理约 `0.199015`；伪组件导致下游 Pin 与 Graph 级联恶化，拒绝该方向并保留当前严格三重门控。
- 这两组结果说明当前问题不是简单扩大 ROI 或回退候选数量，而是需要提升 Component 检测的类别/尺度判别能力，并在 Pin owner 绑定前增加基于几何一致性的候选质量控制。
## 2026-08-27 修复后验收增量

- 主锁 `models/models.lock.json` 已切换为 `candidate_bundle_v1` lineage 的 blessed 可移植发布锁，SHA-256 为 `af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413`；默认配置精确加载 `component`、`ocr`、`pin`、`wire` 四个模型，junction 由确定性 vectorizer 提供。
- 严格命令 `python -m pcbparse.cli healthcheck --require-ready --config configs/pipeline.yaml --models-lock models/models.lock.json` 已返回 exit code 0，文件哈希、bundle manifest、model card、适配器加载和固定图像 smoke 均通过。
- 组件候选级联已加分数门限、NMS 后上限和审计元数据；0024 实图从 300 个原始候选收敛到 14 个组件，OCR/Pin 下游视图均为 14 个；Pin 推理增加 16 项微批上限，语义字段关联改为空间索引。
- 0024 严格 image-only 推理仍正确拒绝发布：当前只唯一解析出 2 个设计标识，12 个组件触发 `COMPONENT.MISSING_KEY`，错误清单见 `work/repair-0024-official.json/error_manifest.json`。这是真实模型/OCR 证据不足，不应通过自动编号绕过。
- 全套测试 513 条：508 条通过、3 条跳过、2 条错误；错误来自冻结训练物化文件的既有 Windows 权限/提交标记问题。Docker 可执行文件在当前主机不可用，因此镜像构建本身仍需目标机复核；Dockerfile fail-closed 契约测试已通过。
- `models/registry.yaml` 与 `configs/models.yaml` 已明确区分 `runtime_ready=true` 和 `production_ready=false`。正式 50-case、3 次重复、P95/显存/离线镜像演练、官方 evaluator 与精度门禁仍是发布阻塞项。

## 历史完成度矩阵（2026-08-23 基线）

| 模块 | 工程状态 | 当前质量证据 | 验收判断 |
|---|---|---|---|
| Dataset300 入库、审计与 split | 已验证 | 300/300 样本；600/600 源图/标注哈希通过；240/30/30 grouped split；147,758 条五任务记录；重复组零跨 split | 已达到当前本地数据门禁 |
| Circuit-IR、坐标链与状态机 | 已验证 | 统一 IR、稳定 ID、3×3 变换、round-trip、状态转换、证据与诊断链均有测试 | 工程完成；仍需随官方 schema 做最终回归 |
| Component | 可运行，未达门禁 | 冻结 val：mAP50 `0.2124`、P `0.3667`、R `0.2338`；门禁 `mAP50 >= 0.25` 失败。val 阈值扫描最优微平均 F1 `0.3018`，阈值无法修复低召回 | 不可 bless；v4 长尾均衡微调排队中 |
| OCR | recognizer 字段证据完整，端到端仍弱 | 冻结 val aggregate exact/CER `0.6785/0.1744`（n=4,277）；designator exact `0.9038`（n=364）、pinnumber exact `0.7569`（n=650）；检测 P/R/Hmean `0.5968/0.3250/0.4208`。0024 冻结视觉键候选为 0/15 exact、0 赋键 | v2c 可作回退候选；detector 和严格 key E2E 未达标，OCR v3 已排在 Component v4 之后 |
| Pin | 旧候选无正式 bundle；v5 正式训练中 | 旧候选现行 decoder：P/R/F1 `0.4249/0.5189/0.4673`、pred/GT `1.221`。v4 在完整 epoch 7 后外部中止，不可登记。冻结 val 的 `boundary_band_v2` 扫描在阈值 `0.75` 得 P/R/F1 `0.8113/0.4496/0.5786`、pred/GT `0.554`，据此启动 v5，但该扫描不是 v5 终态 | v5 自 23:18 运行；中间轮次、smoke 与阈值扫描均不可 bless |
| Wire | 任务级候选通过 | val：skeleton F1 `0.8560`、P `0.8443`、R `0.9845`、connectivity `0.5811`；TorchScript 对齐误差 `0` | 当前最强感知模块；仍未证明最终图拓扑可用 |
| Junction | 真实六类门禁失败 | 六类 macro-F1 `0.4280 < 0.65`；有支持的三类 macro-F1 `0.8560`，但 connected/non-connected cross 与 noise 的真实 val support 均为 `0` | 不可 bless；缺真实标注，不得用 synthetic train-only 标签替代验收 |
| OCR/Pin/Component 关联 | 结构已实现，质量未达标 | 已把 OCR 与 Pin 的 Component ROI 上下文解耦并通过 20 项定向测试；0024 实验仍有 17 个元件缺 key、331 个 Pin 缺 number | 结构修复有效，但语义召回仍是 P0 阻点 |
| Wire/Junction/Graph | 结构已实现，严格 E2E 未达标 | 基线 0024 有 606 segments、689 junctions、110 nets，出现 24 个 disconnected-net ERROR；raw-v1 graph pairwise F1 为 `0` | 图构建可运行，不等于拓扑正确 |
| Validator 与 guide-v1 Serializer | 已验证、正确 fail-closed | 仅在 `VALIDATED` 状态序列化；缺 key/number、断网等 ERROR 会阻断，degraded 结果单独标记 | 工程行为正确；官方格式仍待正式样例/脚本确认 |
| CLI / HTTP | 已实现 | 单图/批量、`/health`、`/ready`、`/v1/infer`、输入根约束和 debug 显式授权均已覆盖 | 调试可用；生产推理受 readiness 正确阻断 |
| 模型锁与发布门禁 | 未就绪 | 五个文件哈希均匹配，精度元数据为 FP32；五项均为 candidate/unblessed。普通 `healthcheck` exit `0` 仅表示存活，`--require-ready` exit `1` | 正确阻止未验证候选进入生产 |
| Docker / 30–50 例演练 | 未验证 | Dockerfile 存在；当前主机未完成目标 Linux/GPU 构建、离线加载、30–50 例总时长及 P95 复测 | 发布阻塞 |
| 官方 evaluator / sample submission | 缺失 | 工作区只有 guide-v1 与本地 raw-v1；源码明确标注它们不是官方 evaluator | 无法给出官方分数或“保证竞赛水平”结论 |

## 数据与证据口径

- 当前机器可读总审计：[completion_readiness_audit.json](../artifacts/audits/completion_readiness_audit.json)。它绑定当前锁、最终回归、五模块指标、训练队列和历史证据漂移。
- 数据审计：[dataset300_audit.json](../artifacts/audits/dataset300_audit.json)。原始数据 0 error、19 warning；warning 为 8 个扩展名/真实格式不一致和 11 个重复/近重复/同工程族组，均已通过内容识别和 grouped split 管控。
- 训练 preflight：[training_preflight_dataset300_v2.json](../artifacts/audits/training_preflight_dataset300_v2.json)。任务记录数为 Component 12,789、Pin 29,648、OCR 48,464、Wire 14,895、Junction 41,962。
- 模型审计：[current_candidate_audit.json](../artifacts/audits/current_candidate_audit.json)。该文件是 2026-08-23 的候选快照；其中 Pin 的旧 lock 参数小节早于现行锁文件，现行参数以 `models/models.lock.json` 为准。
- 严格 E2E：[production_e2e_audit.json](../artifacts/audits/production_e2e_audit.json)。这是临时 benchmark lock 下的单例诊断，不是生产放行或 test 调参依据。
- Tile/ROI 实验：[domain_tiling_roi64_ablation.json](../artifacts/audits/domain_tiling_roi64_ablation.json)。Component 512 / Wire 1024 的域分离方向保留为实验；未提升为 baseline。
- OCR 字段级 val：[ocr_v2c_val_recognition_by_detail.json](../artifacts/audits/ocr_v2c_val_recognition_by_detail.json)；0024 test-only 反事实：[ocr_0024_padding64_diagnostic.json](../artifacts/audits/ocr_0024_padding64_diagnostic.json)。后者明确 `test_only_do_not_tune=true`。
- Pin v5 选参来源：[pin_v4_epoch006_boundary_validation_threshold_audit.json](../artifacts/audits/pin_v4_epoch006_boundary_validation_threshold_audit.json)。它只使用冻结 val，test 记录数为 0；promotion role 为 none。

所有模型指标均来自冻结 `val`，除非明确写为 0024 `test` 单例诊断。0024 的阈值/Tile 结果只用于定位问题，禁止反向选择生产阈值。

## 当前严格端到端结果

当前 baseline 配置仍为 Component/Wire `1024/160`，不会把未通过验证的 512 Tile 实验偷偷设为默认。使用候选 benchmark lock 对 Dataset300 test 样本 0024 做 image-only 严格诊断：

- 感知输出：4 components、66 pins、286 texts、606 segments、689 junctions；单次诊断约 `4.83 s`。
- Validator：87 个 ERROR，其中 `COMPONENT.MISSING_KEY=3`、`PIN.MISSING_NUMBER=60`、`NET.DISCONNECTED_SEGMENTS=24`。
- Serializer：严格输出失败；这是正确的 fail-closed 行为。
- degraded 诊断虽然可稳定写出 JSON，但只剩 1 component、2 pins、106 nets；raw-v1 的 Component F1、Pin F1、Graph F1 均为 `0`，Wire pixel F1 为 `0.8010`。该输出不可视为竞赛可用。

Component 512 / Wire 1024 / OCR padding 0 / Pin padding 64 的解耦实验能检出 20 个元件，但严格 ERROR 仍为 372，且耗时增加；因此只保留架构隔离，不提升实验参数为 baseline。

## 后续执行顺序

旧训练队列已不再代表当前状态：正式模型锁已经包含 Component v5、Pin v5、OCR resume 和 Wire v5，当前主要阻塞转为推理链路、域泛化和发布门禁。

1. **P0：拆分 Wire/Graph 拓扑误连。** 先保留 `wire close=1`、`endpoint_merge=1.5`、`unique_margin=0.5`、`collinear_gap=6` 的 opt-in 候选；继续做线段级置信度、Pin—Wire 归属和大连通块拆分。不要开启 learned junction 或 virtual gap repair，也不要把 Pin max8 候选直接覆盖正式 baseline。
2. **P0：提升语义字段和 OCR owner 链。** 优先解决 Component 的 designator/value、Pin number/name 的定位与归属；Box 标题的 `searchName` 规则已经加入，但当前 exact 增益仍不足，不能代替正式 OCR/字段门禁。
3. **P1：专门化 KiCad 域。** 当前 KiCad 的 Component/Pin/Graph 均显著低于 AD、Datasheet、JLC，应增加域识别、版式归一化和 KiCad 符号/引脚文字专项评测。
4. **P1：恢复严格 E2E 门禁。** 对候选输出执行 Validator → 官方 Serializer 的零 ERROR 复跑；目前 raw-v1 结果只能用于定位瓶颈，不能作为官方分数。
5. **P1：完成发布演练。** 在目标 Linux/GPU 镜像上验证离线权重、`/ready`、30–50 例总时长、P95、显存和确定性；本地 50 例约 13.3 分钟，但尚未完成目标机验收。
6. **外部阻塞：官方 evaluator/提交样例。** 在拿到官方评分器前，继续使用赛会权重的本地 proxy，并明确标注其不是官方 95% 分数。


## 复现命令

最新工作树回归：单元测试 `477` 项通过、`68` 项环境跳过，集成测试 `48` 项全通过；合计 `525` 项通过、0 失败。当前 release-candidate 的 `healthcheck --require-ready` exit `0` 且 ready。

```powershell
$env:PYTHONPATH=(Resolve-Path '.\src').Path
$python=(Resolve-Path '.\.venv-train\Scripts\python.exe').Path

# 1. 三个串行训练配置的数据 lineage
& $python -m pcbparse.training.preflight_cli `
  --config .\configs\train\dataset300_v2_pin_v5_boundary.json `
  --workspace-root . `
  --output .\work\preflight-pin-v5-boundary.json

# 2. 全量测试（Torch 环境）
& .\scripts\run_tests.ps1 -Python $python

# 3. 存活与生产就绪；第二条当前应为非零退出
& $python -m pcbparse.cli healthcheck
& $python -m pcbparse.cli healthcheck --require-ready

# 4. 查看正在运行的 Pin v5；不要另起第二个 GPU 训练进程
Get-Content .\work\train-pin-v5-boundary\torch_run_status.json -Raw |
  ConvertFrom-Json |
  Select-Object status,started_at_utc,finished_at_utc,registerable

# 5. 仅在需要从空目录复现正式 Pin v5 时执行
& $python -m pcbparse.training.torch_train_cli `
  --config .\configs\train\dataset300_v2_pin_v5_boundary.json `
  --task pin --output .\work\train-pin-v5-boundary `
  --workspace-root . --device cuda --precision fp16
```

当前已有两个 fail-closed 串行看护：`scripts/auto_launch_component_v4.ps1` 等待 Pin v5 正式终态，`scripts/auto_launch_ocr_v3.ps1` 再等待 Component v4 正式终态。不要手动并发启动三项训练。

不要手工把 candidate 改成 blessed。最终 promotion 必须由 `data_tools.model_promotion` 使用恰好五个可复核 bundle、Junction 真实六类 human-review evidence、运行时 load/smoke 和新输出路径生成。
