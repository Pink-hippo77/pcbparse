# 可直接派发的 Agent 优化 Prompt

下面每个代码框可以单独复制给能访问本项目的 agent。框中要求它先读取本文件的共同约束。建议先执行 0，再处理 1、2、3；有可信开发基线后启动 4、5；最后由原集成 agent 执行 6。本次审查没有启动其他 agent。

## 共同约束

1. 工作区 `E:\EDA挑战题目六`，历史资料 `archive/legacy-training`。先读适用的 AGENTS.md（如存在）与 `docs/架构阻塞与硬件优化审查_2026-09-08.md`，实际源码、哈希和终态优先于旧文档。
2. 当前机器 i5-12500H、16 GB RAM、RTX 3050 Ti 4 GB。只允许一个 GPU 重任务同时运行；任务开始前重新检查可用内存、显存、磁盘和进程，不擅自关闭用户程序。
3. 项目已有 Circuit-IR、四模型、确定性 Junction、AMP、梯度累积、恢复训练、空间索引和 Pin 全局匹配。先复用，再修局部问题。
4. 生产推理只读图像，不读对应标注。GT、oracle 和人工真值只能进入离线评测/训练，明确记录输入边界。
5. release50 包含全部 30 张原 test 和 20 张 val，已反复用于选参；保留清单与曝光历史，按已曝光回归集管理。不能再声称独立 test，也不能重新抽取旧数据伪装未曝光 holdout。
6. raw-v1、guide-v1、strict 结果分开报告。几何 F1 不等于官方 Component/Pin F1，normalized 文本不等于严格文本，pairwise Graph 不等于 net assignment。
7. 不删对象、不制造 UNKNOWN key、不放宽 Validator 来伪造成功。官方允许的重复 designator 后缀和空 pin number 转换必须保留，并记录原始字段与转换来源。
8. 审查时主锁 SHA-256 为 `af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413`。开始时重新核对；不手工修改 blessed，不覆盖原模型、原始数据和现有运行目录。新候选、配置、报告写新路径。
9. 主目录审查时没有 Git。先由集成 agent 建立版本化基线；已有仓库时使用隔离分支/工作区。一个文件同时只有一个写入负责人，其他 agent 通过建议或补丁交接。
10. 0 和 6 由同一个集成 agent 负责。1 负责图与图校验；2 负责元件命名与 serializer；3 负责 runtime/资源；4 负责 OCR 与文字归属；5 负责 Component/Pin 数据和模型。`pipeline/orchestrator.py`、`evaluation/infer_manifest.py`、默认配置和主锁由集成人员统一修改。
11. OCR adapter、Composite provider 等共享文件按阶段交接，避免同时编辑。GPU 任务串行；CPU 轻量分析可有限并行，先保证可用 RAM。
12. 不复制本地凭据、API key 或整个 work 目录到交接包。不得自动调用付费 API、上传图纸或发布外部内容。
13. 每次交付写明：修改文件、原因、实际命令、结果与分母、退化、峰值资源、源码/配置/模型/数据哈希、剩余阻塞、回滚方法。未执行检查明确标记。
14. 不以“已分析”“建议继续”结束任务。完成所负责范围内的代码、必要测试和有界实验；外部条件缺失时交付可运行入口并明确缺项。

## Prompt 0：建立可信基线与评测入口

```text
你负责本项目的集成与评测基线。工作区 E:\EDA挑战题目六。
先读取 docs/Agent优化执行Prompt_2026-09-08.md 的共同约束和
docs/架构阻塞与硬件优化审查_2026-09-08.md，然后落实以下任务。

目标：让后续 agent 的结果可比较、可追溯，并恢复严格失败的可回放证据。

先检查：
- models/models.lock.json、configs/pipeline.yaml、configs/pipeline.release-candidate.yaml；
- data/splits/dataset300_split.jsonl、dataset300_release50_v1.jsonl；
- src/pcbparse/evaluation/official_metrics.py、cli.py、infer_manifest.py；
- src/pcbparse/pipeline/orchestrator.py、现有 release audit 工具。

实施：
1. 若没有 Git，建立本地版本化基线，明确排除环境、数据、权重、work 和凭据；
   不上传远端、不覆盖其他人的改动。记录启动时全部有效配置和模型来源。
2. 生成新的 authoritative_status.json：记录 runtime-ready、strict-ready、release-ready
   三种状态、四个实际模型、已运行/失败/未运行证据。OCR 实际是旧 resume；OCR v8 和
   Component v6 均已 gate_failed，不能照抄旧文档中的“等待训练完成”。
3. 审计 split 曝光历史：release50=30 test+20 val。保留原文件，新增曝光登记和
   开发/回归/独立 holdout 角色说明。新独立数据未到位时继续工程工作，不伪造独立性。
4. 建立统一实验汇总入口，复用 raw-v1 与 evaluate_official_json；同时报告：
   几何诊断、严格字段、GT 可评估覆盖、guide-v1 逐案例分数、strict JSON 成功率。
   guide-v1 保留 micro 与固定案例分母下的 macro；失败案例不能从分母消失。
   expected JSON 不可构造时标记覆盖不足，不将局部可评估子集包装成完整赛会成绩。
5. 让严格失败也保存同一次运行的 IR、原始 diagnostics、阶段 trace、源码/配置/模型哈希；
   failure IR 不标成 VALIDATED，不生成 official JSON，不消除失败状态。
   避免把完整 IR 放进异常字符串；使用受控的本地 failure artifact 接口和原子写入。
6. 用一个合成失败和一个已有缓存案例验证失败 IR 可回放。后续开发基线只跑一次，
   保存可复用感知缓存；参数实验不重复跑全部模型。

验收：
- 错 config/model hash、缺案例、仅成功子集、缺 expected 真值不能被报告成完整成功；
- 严格失败有可读取 IR 且 status 仍为 error，成功/失败两类都绑定来源；
- production eval、release gates、官方指标定向测试通过，新增测试覆盖上述真实边界；
- 输出 docs/optimization-20260908/baseline.md、新 audit JSON 和复现命令。

使用 .venv-test\Scripts\python.exe 跑轻量测试；真实模型用 .venv-train。
你拥有 orchestrator.py、infer_manifest.py 和评测集成入口的写入权。
本任务不更换模型、不调整几何阈值，不执行 3×50 发布演练。
```

## Prompt 1：修复 Physical Graph 与 Validator 的连接契约

```text
你负责图连接正确性。工作区 E:\EDA挑战题目六。
先读 docs/Agent优化执行Prompt_2026-09-08.md 共同约束与对应架构审查。

首要代码：src/pcbparse/graph/physical.py、validation/graph.py、
association/pin_wire.py、tests/unit/test_graph_fixtures.py、test_ir_graph.py。
先执行 artifacts/audits/architecture_review_20260908/reproduce_contract_findings.py。

已复现的三个问题必须在修改前确认：
A. y=10 的 [0,20] 与 [10,30]，endpoint=1、collinear overlap=.75，
   构建为一个 net，Validator 却报 NET.DISCONNECTED_SEGMENTS。
B. [0,10] 与 [15,25]，gap=6 可被构建器合并，Validator 不接受；
   不能为消除 ERROR 无条件允许跨 5px 空白连线。
C. 两条相距 40px 的预测线段复用 source_polyline_id，构建器不合并，
   人为放到同一 net 后 Validator 却放行。

实施：
1. 提炼版本化连接契约，分别定义端点接触、同通道共线重叠、T/十字语义连接、
   有像素/局部修复支持的间隙连接、明确来源的 golden 连接。
2. 构建与验证共享基本几何判定和参数定义；Validator 必须独立重算几何/证据，
   不得仅信任 builder 的 union ID 或 claimed_connected 标志。
3. gap 修复需要真实证据与限额；保留 non-connected cross、平行近线、
   无证据间隙和远距离复用 ID 的负例。不要启用 learned Junction。
4. 在缓存 IR 上离线重建并对比：split/merge、最大 net、无 pin net、
   disconnected-net、Graph precision/recall 与错误 pin pair。
   区分“校验假阳性修复”和“网络质量提高”，两者不能混为一个指标。
5. 如需传递 config/contract 到 orchestrator，先给集成 agent 清晰接口补丁，
   由其统一合入；保持旧 IR 读取兼容，不信任旧 IR 缺失的证据。

验收：合法共线重叠不再误报；无证据远距离连接被拒绝；crossing 测试保持；
构建与独立校验在合成契约集一致。缓存回放仅称诊断，不冒充新 production run。
不靠删掉 net/segment 或统一放大容差达标。

交付实现、最小反例、定向测试、带前后 TP/FP/FN 的 replay 报告和回滚补丁。
先用 .venv-test 执行 CPU 检查，本任务不训练、不占 GPU。
```

## Prompt 2：统一可见标签、元件 key 与合法序列化编号

```text
你负责元件命名契约。工作区 E:\EDA挑战题目六。
先读 docs/Agent优化执行Prompt_2026-09-08.md 共同约束和架构审查。

阅读：association/text_component.py、association/ocr.py、validation/schema.py、
output/official_serializer.py、output/official_schema.py；
官方指南 artifacts/reference/2026_EDA_Challenge_Task6_AI_PCB_Schematic_Official_Guide.pdf
第 14–18 页；tests/unit/test_ocr_association.py、test_validation_serializer.py。

目标：让正确视觉文本进入官方 key，而不是因旧正则/唯一性规则被拒绝。
当前反例：明确归属的 RUN 被拒绝；两个元件分别有明确 owner 的 R1 文本，
最终只有一个得到 key。Validator/Serializer 本身已支持重复 designator 后缀。

实施：
1. 用独立内部 ID 表示元件实例，保留原始可见 designator，在输出阶段生成唯一 key。
2. 区分同一 token 被多个元件争抢，与两个元件各自观察到相同文本。
   前者继续处理歧义，后者允许相同 designator 并稳定输出 KEY_1/KEY_2。
3. 对 RUN/VCC/GND 等无数字文本，要求合法类别、明确视觉上下文与 owner；
   不把任意字符串放宽为合法 key，不把 net label 强行变成元件。
4. 重复后缀和 pin 自动编号由统一 serializer 负责；保留原始编号及来源，
   禁止覆盖真实可见 pin number。输出排序与官方自然排序约定一致。
   原图信息无法恢复 GT 内部 ID 顺序的场景单独记录，不保证后缀恰好匹配。
5. 清理严格 serializer 中缺 key 时制造 UNKNOWN_* 的兜底：严格调用应明确失败，
   不修改输入 IR。不要通过手动设置 VALIDATED 隐藏缺失字段。
6. 用缓存文本复核不同类别和重复标签，报告 key coverage、严格 key exact、
   错误 owner、冲突数与未解决项，不能只报告 missing key 减少。

验收：RUN 和有独立 owner 证据的重复标签正例通过；单 token 抢占、后缀碰撞、
无证据 key、空白文本、真实 pin 编号冲突等负例仍失败；置换输入顺序后输出稳定；
serializer/命名/关联定向测试通过。给出新规则来源和适用范围。

文件所有权：text_component.py、validation/schema.py、official_serializer.py 及测试。
semantic_fields.py 如需变化，交接给 OCR agent，避免同时编辑。
不训练，不覆盖正式配置/主锁；交付规则实现、复现、回放审计和回滚方法。
```

## Prompt 3：修复 CUDA 依赖并落实 4GB 资源策略

```text
你负责本机 runtime 和资源优化。工作区 E:\EDA挑战题目六。
先读 docs/Agent优化执行Prompt_2026-09-08.md 共同约束和架构审查。

确认现状：RTX 3050 Ti 4GB、16GB RAM；Torch 2.12.1+cu126 / CUDA 12.6；
onnxruntime-gpu 1.28.0 需要 CUDA13。Pin _ONNXBackend(device='cuda')
仍可创建 CPU-only session，缺 cublasLt64_13.dll；available 列表不能证明实际执行。

实施：
1. 保留 .venv-train，在新 runtime 环境验证 CUDA12/cuDNN9 兼容的 ORT 构建。
   优先核实 ORT 1.26.x 精确可用版本、Python wheel 和 ONNX 兼容性，再固定版本。
   依据 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html，
   不下载来源不明 DLL，不默认升级全部 CUDA/Torch。
2. 修改 perception/torch_pin_adapter.py：显式 CUDA 请求但 session 未启用 CUDA 时失败；
   auto 回退记录原因。记录 requested/actual device、session providers 和版本。
   运行真实模型小批次，用 ORT profiling 验证主要算子位置，不能只列 provider。
3. 将 Pin 写死的 microbatch=16 改为有界配置；以 1/2/4/8 测峰值和吞吐。
   总显存包含所有模型、ORT 和其他进程，不能只看 torch.max_memory_allocated。
   对接集成 agent 的 runtime evidence，补充按模块设备和峰值。
4. 建立 GPU 重任务租约/心跳，防止 stale running 与多个 watchdog 重复启动。
   不杀用户进程，不扫描进程名后直接结束所有 python.exe。
5. perception 占历史阶段耗时约 92.5%；先添加 Component/OCR detect/OCR recognize/
   Pin/Wire/vectorize 子阶段计时。Torch/ORT CPU 线程试 4/6，workers 先 0 后 1/2。
   OCR 宽度分桶和 ROI 调度如涉及共享文件，与 OCR agent 串行交接。
6. 统一 runtime lock 与 Docker CUDA 主版本；导出一致性验证前不替换正式环境。

验收：显式 CUDA 负例失败且可解释；正常 CUDA 有算子执行证据；
CPU/GPU 输出符合既有导出误差契约；microbatch 改变不破坏候选顺序/内容约定；
报告冷启动、P50/P95、RSS、显存、实际线程/批次，不承诺未测的加速倍数。
必要检查包括实际 Pin 模型、production provider/readiness 和 Docker contract 测试。

交付精确依赖清单、版本理由、资源配置、诊断工具和前后对比。
若资源不足，受控降批次并记录，不能把 CPU 回退冒充 GPU benchmark。
```

## Prompt 4：修复 OCR 到字段与 owner 的信息损失

```text
你负责 OCR 与语义字段端到端质量。工作区 E:\EDA挑战题目六。
先读共同约束、架构审查和集成 agent 的开发基线。
依赖 Prompt 2 的命名接口，不与其同时编辑 text_component.py。

阅读：perception/ocr_multilevel.py、torch_ocr_adapter.py、association/semantic_fields.py、
text_pin.py、pin_assignment.py、ocr_common.py、training/torch_ocr_model.py。
复用 scripts/audit_ocr_recognition_by_detail.py、audit_ocr_detector_threshold.py、
analyze_semantic_error_pairs.py；不要重做已有 Pin 全局匹配。

已知：现行锁 OCR resume exact=.5307、detector Hmean=.3669。
OCR v8 已完成但 Hmean=.44583 未过 .45，不能直接晋级。
已曝光 release50 的 designator exact=114/650、pin number=72/1508，
仅作诊断参考，不可继续用它挑生产阈值。

实施：
1. 建立开发集损失分解：文字框未检出、识别错误、语义类型错误、owner 不存在/错误、
   组选择错误、匹配拒绝、字段冲突、最终 exact。保留少量 overlay 和 top-k 候选；
   GT 缺字段单独 mask。
2. 复用缓存做逐层 oracle 替换，仅供离线诊断，把 GT 输入与真实推理隔离。
   确定主损失在 detector、recognizer 或 owner 后再选择改动。
3. 检查全图/元件/侧边 OCR 的重复计算、padding、小字缩放、相邻 ROI 污染及去重。
   用文字几何增强 owner 证据；ROI 来源只作先验，不能使错误 ROI owner 不可纠正。
4. 短文本按宽度分桶减少 padding，恢复原稳定顺序；批次 8/16 起测。
   如做按不确定性触发的二次 OCR，保留全扫描对照，不能靠少读文字损害召回。
5. 若需要训练，用新路径与明确 --task ocr 启动一项实验；复用 AMP、累积、checkpoint。
   增加来自 train 的真实困难文本/负样本；禁止把 val/test 或其 crop 混入训练。
6. 新 OCR candidate 沿用 v8 四门：exact>=.65、Hmean>=.45、
   designator exact>=.90、pinnumber exact>=.74；另提供 owner/字段 E2E 证据。
   未过门禁时解释损失，不放宽门禁或盲目延长 epoch。

验收：同开发协议下字段 exact/召回提升；错误 owner、Graph 退化和语义误分类单列；
golden 不受污染；密集 pin、旋转侧边文本、重复 token、无 owner、冲突定向测试通过。
给出各层损失与资源消耗，由集成人员决定候选晋级。
交付实现、配置/候选（如有）、命令、失败样例和回滚方案。
```

## Prompt 5：Component/Pin 联合召回与域专门化

```text
你负责 Component/Pin 数据与模型质量。工作区 E:\EDA挑战题目六。
先读共同约束、架构审查和开发协议，GPU 资源与 OCR agent 串行。

阅读 perception/composite_provider.py、torch_component_adapter.py、torch_pin_adapter.py、
training/component_pin_materialize.py、torch_component_model.py、torch_pin_model.py；
复用已有 stratified audit、tiled validation、oracle decomposition。

目标：减少组件漏检导致的 Pin/Net 丢失，解决预测 ROI 与训练 GT ROI 的分布差异。
现有 KiCad 诊断只含 2 张，Component/Pin/Graph F1 约 .413/.168/.026，
是薄弱信号，不足以证明整域泛化结论。

实施：
1. 按域、组件大小/类别、密集度、旋转、边界、pin 数建立开发集矩阵；
   统计 Component 匹配率、匹配组件条件下 Pin recall、正确 owner recall。
   检查 component_max_candidates=128、每组件 Pin max12、缩放和 NMS 的截断损失。
   不直接调大所有上限，先区分高 pin 数 IC 与简单元件的假阳性/漏检。
2. 对比 GT ROI、抖动 GT ROI、预测 ROI 上的同一 Pin 模型。
   预测 ROI 训练素材只用 train 内 grouped out-of-fold 元件预测产生，
   附源码/模型/框/坐标 lineage。
3. 若 ROI 差异是主损失，加入框抖动/边界目标、预测框 hard negative，
   必要时按类别/边分配候选数，保留局部/全局坐标可逆性。
4. 对 train 中 KiCad/小目标做数据质量检查和定向增强。
   域路由依赖图像特征，不准使用标注、样本 ID 或 test 文件名作 oracle 路由。
   先验证 512/256 局部分流、字符/符号尺度归一化，再考虑新架构。
5. Component batch=1；Pin batch=1–4 起步；有效 batch 用现有累积维持。
   Component 保留 FP32 基线，FP16 先检查数值/导出一致性。
   每轮只改一个主因素，明确 --task，避免旧五任务模板全部启动。

验收：同开发集下 Component、Pin、owner、Graph 和字段同时报告；
不能只用 PCK/mAP50 宣布成功。记录各域样本数与不确定性。
训练/物化不跨工程族泄漏；未覆盖的真实类型不以合成验证样本补齐。
新候选使用新 bundle 和新路径，由集成人员验收，保留当前锁。

交付损失分解、数据清单、实现/配置、一次有界对照、峰值资源和回滚方法。
不顺带进行无关的大模型重写。
```

## Prompt 6：冻结候选、统一入口并完成发布证据

```text
你是执行过 Prompt 0 的集成 agent。工作区 E:\EDA挑战题目六。
先读共同约束、各任务交付和实际源码；在前述候选可验收后执行本任务。

目标：组合已证明有效的改动，完成真实严格交付证据。

实施：
1. 按文件所有权逐项集成，解决接口/配置冲突。保留四模型+确定性 Junction；
   模型替换仅通过现有 promotion/bundle 工具写新锁，不手工改 blessed。
2. 统一版本化配置绑定权重、推理覆盖参数、图契约和评分协议。
   CLI、HTTP、Docker 加载同一配置；当前 Docker 的 pipeline.yaml 与
   release-candidate 不同参数不能静默混用。
3. 复用 guide-v1 matcher 和严格 Serializer，按固定案例分母报告分数、GT 覆盖、失败数。
   模型 smoke 要覆盖四个任务，不能只用零组件/零 Pin 的空白图。
4. 先跑冻结开发小套件；存在系统性 missing key/disconnected-net 时保留失败 IR，
   退回责任模块，不重复执行无收益的 3×50。
5. 候选冻结后 canonical release50 连跑三次同环境 image-only 严格推理：
   每次 50/50 产出可校验 official JSON、0 ERROR，逐案例 JSON/IR 语义哈希一致。
   报告冷启动、全流程、P50/P95、实际设备、峰值 RAM/显存、调用与 token。
   7200s 硬时限按现有约定保留 10% 余量；缓存回放不能算完整模型推理。
6. 现有 coordinate、trace、determinism、hard-gate 工具消费同一次正式 run；
   不以新源码重新包装旧 summary，不混用 lock/config。
7. 修正 Docker CUDA/ORT 兼容性；.dockerignore 覆盖 .venv-test、.venv-train、
   临时目录和无关训练产物。记录镜像版本、模型哈希。
   目标 Linux/GPU 验证离线运行、只读输入、错误码、单推理槽、/ready、失败恢复。
   本机无 Docker 时交付构建/测试入口和未验收状态，不假称通过。
8. 新独立 holdout 或官方 evaluator 到位后再做独立确认与官方适配。
   保留 release50 曝光事实，官方评分器缺失不能由本地 proxy 替代。

交付：必要单元/集成/模型检查结果（skip 说明原因）；新锁、新配置、版本一致的状态报告；
提交包、依赖清单、复现命令、三次结果及目标机证据（已具备时）。
未通过的硬门禁保持 false；报告改善、退化、数据边界与回滚路径。
不承诺赛会95%或未知官方分数。
```

## 派发顺序与统一交付格式

先完成 0 的基线，再优先投入 CPU 即可验证的 1、2，以及恢复实际 GPU 执行的 3。随后依据损失分解决定 4、5 的 GPU 时间占比。1、2 的文件边界独立，但内存不足时仍应串行。共享文件或同一 GPU 上的任务不能同时写入/运行。

每个任务交付：代码版本、配置/模型/数据哈希、样本分母、关键指标前后值、失败类别、耗时与峰值资源、已知退化、剩余依赖、回滚方法。性能和精度收益以实际测量为准。
