# 当前系统性能瓶颈检索与优化方案

日期：2026-10-03。实验与报告均在服务器 `/home/data/yinjx/pcbparse-competition/project` 生成。

目前应优先优化 OCR 解码、局部文字读取和字段关联，再处理引脚几何与网络拓扑。已有 YOLO26l、PP-OCRv6 medium 和更宽 Pin32，继续扩大模型不能直接解决缺少器件 key、引脚编号或错误连接的问题。以下区分本系统实测结果与需要验证的文献方法；没有声称达到四项 95%。

## 本系统的实测证据

OCR 画像采用 2 张训练图的 8 个放大器件 ROI，共 376 个文字裁剪、26 个识别批次。原始 OCR 推理合计 18.399s，其中 CTC 解码 10.377s，占 56.40%。这是该样本的 CPU 解码瓶颈，不能推断所有图均有相同比例。

cycle76 私有原型以 argmax 取得索引，再按索引收集分数，并保留维度、有限数值和概率范围检查，减少 18,710 类概率张量的反复扫描及临时数组。这一索引收集方法有 [NumPy 2.3 官方说明](https://numpy.org/doc/2.3/reference/generated/numpy.take_along_axis.html)。在同一组真实概率上交替执行原实现与候选：9.1723s → 0.2892s，解码加速 31.71 倍；26 批文字及置信度 JSON 字节完全一致，8 个最终 OCR 输出哈希一致。33 个边界/错误合同用例通过。这里的倍数仅指解码，整套流水线已通过下述 cycle79 新推理验证；独立的纯解码因果加速比例仍不据此推断。另一轮 OCR 总耗时跨运行下降约 61.5%，受预热及共享 CPU 负载影响，不作为隔离的加速结论。

局部 scale2 OCR 的 cycle74 在冻结原器件、引脚及导线感知结果上重新运行字段绑定至严格输出，30 个控制输出与 cycle57 一致，候选有 20 张改变；总计 62 次严格序列化均为 0 ERROR，2 个候选重复输出字节一致。以下是 **raw-v1 开发诊断**，不是 ComponentF1/PinF1/NetHypergraphF1/NetLineF1 正式成绩：

| 开发指标 | cycle57 | cycle74 | 变化 |
|---|---:|---:|---:|
| 器件框 F1 | 48.87% | 49.04% | +0.16 个百分点 |
| 引脚点 F1 | 31.86% | 33.82% | +1.96 个百分点 |
| 成对连接 F1 | 3.75% | 4.27% | +0.51 个百分点 |
| 线像素 F1 | 50.49% | 50.49% | +0.00 个百分点 |

严格匹配字段：器件编号 242 → 240；器件型号 10 → 10；数值 131 → 131；引脚名称 184 → 201；引脚编号 133 → 150。局部 OCR 有收益，但损失 2 个器件编号，不能直接晋级。所有新增绑定均来自实测 OCR 原文，无自动编号、覆盖非空字段或 GT 读取。 比较纯图像生成的 association IR 还发现 `0030` 中 key 从 `R3` 变为 `LM334`，`0038` 从 `U5` 变为 `PCF8563`，新 key 置信度均超过 0.9999。源码核查还发现实验 `pair_features.valid_role` 虽接收 category，designator 实际只检查通用字母加数字正则，未使用类别参数，`LM334`、`PCF8563` 均可通过。这提示“器件型号原文被当成 designator”的角色竞争需要优先审计；12 个 key 变化不全是错误，不能仅凭变化数宣称回退。这里比较的是不同运行的字段结果，并非在一次运行内覆盖已有非空字段。严格序列化通过不代表输出完整；当前仍有缺字段导致的裁剪。

cycle74 线像素召回 94.37%、精度 34.46%，需要先审计多余线像素来源。成对连接仅 79 TP / 364 FP / 3181 FN，说明缺失可用引脚和连接错误都需处理；该结果尚不能证明交叉线误合并是唯一原因。现有 Physical Graph 已有空间网格和 junction 分类，cycle16 空间搜索、cycle20 mask 缓存也已验证，不能把重复实现它们算成新优化。

## 与赛题指南的对应关系

官方指南要求：器件按 key 匹配，并检查 type、bbox 默认 20px 和必要文本严格一致；引脚按 component key 与 pin number 匹配，名称正确、坐标默认 5px；网络按 pin set 匈牙利匹配；线段只在匹配网络内按两端点默认 5px 比较。因此，上游编号错误会影响多项，线像素召回也不能代替 NetLineF1。每张 600s、整批 7200s 的约束保持不变。

指南原文保存在 `work/optimization-cycle22-server-parallel-20261002/official-guide.txt`（实际路径以本报告证据清单为准）。官方 evaluator 仍未获得；raw-v1 用于诊断，guide-v1 只作指南回归。四项均 ≥95% 的验收条件不变；已反复使用的 val/test/release50 不宣称独立泛化。

## 检索后建议的实施顺序

| 顺序 | 方法 | 对应瓶颈 | 验证与退出条件 |
|---|---|---|---|
| 0 | 快速 CTC + 局部 scale2 OCR 整套集成 | 已验证解码开销与漏字 | cycle79 四模型新推理 30+2 已通过，30 张与 cycle74 字节一致，单图与开发批次均在预算内 |
| 1 | 器件 key / 引脚编号绑定加入侧边对齐、跨角色竞争与局部唯一性审计 | 原文存在但绑定缺失，且局部 OCR 损失 2 个 key | 先定位 2 个回退例；保持原文，训练内部划分校准；报告每项字段和各开发指标，不能只报告总分 |
| 2 | 文字掩膜后骨架端点与器件边界相交，结合引脚热图与方向 | Pin32 实际 ROI 漏检及图线关联不足 | 几何必须由图像支持；5px 端点、owner、编号、名称分别评估；无 GT 或器件库自动造编号 |
| 3 | 多余线像素及交叉/连接点分层诊断，再改方向保持的 Physical Graph | 网络精度低、召回低 | 分开统计缺 pin、错接、断线、线框/文字混入；保持交叉方向轨道；同时检查网络集合与所属线段端点 |
| 4 | ORT 算子画像后，GPU 端只输出索引/分数/数值有效标志，并绑定 GPU I/O | 大字典概率输出传输与剩余 Session.run 开销 | FP32、有限值错误合同及 OCR/final JSON 一致；包含传输同步耗时，不能只计 kernel |
| 5 | 长宽分桶、稳定缓冲区，再评估 CUDA Graph / 高性能引擎 | 动态尺寸与 GPU 启动开销，收益尚未测得 | 同输入、同预热、真实 provider 分区；兼容已安装 ORT1.21.1；不因文档示例盲目升级环境 |
| 6 | YOLO26 的小目标分层切片调优 | 剩余小器件漏检 | 当前已切片，先定位 tile 边界/小目标漏检；同时看重复框、字段和时间，再决定是否需要更大权重 |

[PCBnet 原论文](https://arxiv.org/html/2608.27923v1)支持局部文字检测再识别、竖排文字旋转，以及类别前缀、对齐、唯一性约束。其 PaddleOCR 整图 ACC 78.65%，检测辅助 93.67%；这是作者自己的 OCR 指标。本系统只借鉴视觉与约束方法，不采用其 LLM/VLM 补字模块。公开 210 张与现有 Dataset300 重复，不能作为新增 holdout。

[SINA 原论文](https://arxiv.org/html/2607.01609v1)采用文字掩膜、骨架端点与膨胀器件框相交来找 pin，并在网络合并前分离交叉轨道。其 Skeleton-OCR 引脚几何 F1 为 95.3%，不是赛题包含 key/编号/名称的 PinF1。可借鉴几何与交叉消歧；本项目只在 Circuit-IR/Physical Graph 内保存方向关系，不生成论文中的虚拟器件到官方输出。

[ORT 官方 profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)可记录算子延迟与线程事件；provider 列表出现 CUDA 不能证明所有算子实际在 GPU。[I/O Binding 文档](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)说明默认 CPU 输出存在设备复制。绑定后若仍复制完整概率张量到 NumPy，收益会受限；GPU 端缩减输出是需要验证的工程推断。

[CUDA EP 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)要求 CUDA Graph 的全部算子可分到 CUDA，输入输出形状和地址固定、使用 I/O Binding，同一 Session 不可多线程 Run；首次捕获不计稳态收益，但需要计入实际批次时长。识别宽度分桶可先帮助稳定形状，这是本系统的实验建议，尚无加速结论。FP16/TensorRT 作为后续候选，需要重新检查文字、置信度和输出，不据文档推断兼容已安装 v6 模型。 [PaddleOCR 高性能推理官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/inference_deployment/local_inference/high_performance_inference.en.md)也注明引擎支持与模型算子、静态图和 CUDA 环境有关；不能把 `enable_hpi` 当成当前 ONNX adapter 可直接打开的开关。

[Ultralytics YOLO26 + SAHI 官方指南](https://docs.ultralytics.com/guides/sahi-tiled-inference)提供高分辨率分片与重叠合并；本系统已有 tiled inference，应做增量对照，不能把示例 256px/20% 重叠当成该数据集的最优参数。

## 服务器执行与模型保留

当前 104 逻辑 CPU，503GiB RAM，约 398GiB 可用；共享负载约 164，已经高于逻辑 CPU 数。GPU0 A6000 正被其他任务使用；本项目只用 GPU1 A6000 48GB。此前 8×2 CPU 线程相对 4 进程整批观察耗时下降 21.04%，输出一致。继续采用有界并行，不因 CPU 多而无限增加线程。GPU lease、UUID 核对、显存预留和失败关闭保持。

cycle79 四模型新推理已经完成：30 张 + 2 次重复，32 次严格序列化全部通过，30 张官方格式 JSON 与 cycle74 字节一致，重复 JSON 也一致。开发批次观察耗时 453.57s，最慢单图 323.87s，显存峰值 34,557MiB；这是完整推理加证据输出、预热和调度的时间。相比 cycle58 同为 8 进程的 483.98s，观察下降约 6.28%，但新运行增加了局部 OCR 且共享负载不同，不能归因成纯 CTC 加速比例。此处只验证 30 张开发批次，不外推全比赛批次。只读图像、现有模型、配置和静态器件库，GT 评估在实际推理进程全部退出后进行，已经完成且与 cycle74 的 raw-v1 汇总完全一致。所有本轮实际 worker/controller/evaluator PID 已退出，模型与生产源哈希已重新核对。没有新增权重或依赖，没有修改生产代码、配置或模型锁。cycle70 新 pin role 模型没有输出收益，cycle77 已按哈希和实际进程检查删除其 53,457 字节权重，保留证据；此删除量很小，不夸大为显著磁盘收益。保留现用 YOLO26l、medium OCR、Pin32 及必要回滚/恢复资产。

此前关于 native Pin budget 坐标单位错误的判断已撤回：实际 helper 返回 inference-canvas 坐标，native budget 调用正确，16 个相关单测通过。cycle65 为错误假设实验，不能再列为当前瓶颈；详见 corrected checkpoint67。Pin32 原训练晋级门禁未通过，所有这些结果仍是开发候选。

当前工作顺序：cycle79 新推理与离线汇总均已通过，final raw-v1 全部结果等于 cycle74；接下来诊断 key 回退与字段绑定，随后处理 pin 几何及网络误接。只有候选在四项指南口径、严格 Validator、确定性、lineage 和时间约束上有完整证据，才能使用正式晋级工具；现在未晋级，验收未完成。
