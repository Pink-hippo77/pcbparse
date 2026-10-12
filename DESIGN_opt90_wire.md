# DESIGN — opt90-wire：wire 分割模型笔画监督重训（goal 线 B）

日期：2026-10-08 · 线程：wire-model-tune · writeScopes：`work/opt90-wire/`、`configs/train/opt90_wire_*.json`（只新增）
基线口径：r12-integration 终局（val30，qna-v1 本地代理口径，非竞赛成绩）：wire **P 0.6911 / R 0.8659 / F1 0.7687**，graph pair **TP 25 / FP 8**。

## 0. 问题定义（为什么不走规则）

r12-wire 四层归因（`work/module-opt80-20261007/r12-wire/REPORT_round6_wire.md`）：
wire FP 96 457 px 中 **86% = 图纸装饰（图框/分区隔断/标题栏表格）+ 器件符号笔画（芯片体外框、引脚方框/短桩梯、电容极板、地符号）**；
确定性规则（`filter_decorative_wire_segments` 四签名）已榨干可分结构（FP -47% → 51 131 px，P 0.6911），
残余 FP 与真导线在全部 GT-free 特征上不可分（单特征 Youden J ≤ 0.41；颜色、置信度同分布）→ **model-bound**。
结论：要跨 0.90 只能让分割器本身不再把符号/装饰笔画判为 wire。

**训练监督缺口（本设计的核心动机）**：现行 wire 物料化（`wire_junction_materialize`）的 ignore 语义是
"组件 bbox（margin 0）+ 文本 bbox + pin 线（宽 3）→ 不监督"。而 FP 主体的**符号笔画恰好住在组件 bbox 内/边线上**
（interior 签名抓到的 254 段、rect_outline 242 段均在 bbox 内/边缘）——即**模型从未被要求对这些笔画输出负样本**；
图纸装饰虽是监督负样本，但在均匀网格采样下相对整图背景占比极低，梯度贡献小。

## 1. 负样本标注器（离线，train240）— 2026-10-08 预注册修订（结果前，Lead 已批准）

> **修订要点**：四签名标注器原样迁移、产出"符号/装饰笔画"区域与误伤率（任务书要求，不变）；
> 但 **训练负样本 NEG 定义为现役模型的误检像素**。理由（证据 r12k/r12b）：四签名已被推理期规则使用，
> 其命中的是规则已抑制的 47% FP；要跨 0.90 必须消灭的 **残余 51k px 恰恰是四签名抓不到的无彩色符号笔画**。
> 因此 NEG = 残余 FP 的同分布监督对象，SIG（四签名）降级为分层与误伤核验工具。

**NEG 口径（写死，防口径漂移）**：
train240 每张源图上，**生产 wire 模型**（models.lock wire entry）按**生产推理路径**
（`tile_image_view` 生产 tiling → `TorchWireAdapter.segment` → 生产 `_vectorize_masks`）输出的
**误检像素**：模型 fired mask 中 **距任意 GT wire 中心线 > 3px**（tolerance 与 masked_metrics qna-v1 口径一致）的像素。
GT wire 几何 = `wire_paths.jsonl` 的 keyPointPaths + skeleton（与训练 target 完全同源）。

**可复现命令与哈希**：
```bash
cd /home/data/yinjx/pcbparse-competition/project
PYTHONPATH=src <runtime python> work/opt90-wire/opt90_neg_labeler.py --workers 24
```
- wire 模型：`models/releases/v8-portable/wire/train-wire-v5-fresh-allgate/model.pt`
  sha256 `3235b9e083190e8d399f819b55a34ed8531652377323ff8be05ae99286ea17dd`（models.lock wire entry，blessed）
- models.lock sha256 / pipeline config（`work/module-opt80-20261007/r12-integration/pipeline-integration.yaml`）sha256 /
  `wire_paths.jsonl` sha256 `6deb93c959da75c26a4c93657e924e9785c52cbb11ab91b33128a96185c6438c` /
  标注器脚本 sha256 —— 全部记录在 `neg_stats.json.method`（运行时实测），任一不符即重跑。

**SIG（任务书要求的四签名迁移产物）**：生产 `filter_decorative_wire_segments` 跑在同一套模型笔画 segment 上
（GT 组件 bbox / GT 文本点做 far-gate），产出"符号/装饰笔画"区域标注 SIG（笔画栅格 dilate 2px，保护前原始版），
并按 canvas_span / companion / rect_outline / interior 分层统计 NEG 覆盖。

### 误伤率量化协议（进训练前强制）

- `sig_px`：SIG 像素数；`sig_hit_px = |SIG ∩ dilate(GT_wire, 3)|`；**SIG 误伤率 = sig_hit_px / sig_px**；
- **预注册门槛：SIG 误伤率 < 2%**（超过 → 收紧签名参数后重新量化，达标才进训练）；
- 训练 NEG 同样报告保护前命中量 `model_near_miss_px = |model_mask ∩ dilate(GT_wire,3)|`（模型近失质量，如实入档）；
- **注入时硬保护：NEG_eff = NEG − dilate(GT_wire, 3)**，保护带内保持原监督语义（NEG 与 GT 构造性不相交）；
  物料化断言：① `target ∩ ignore = ∅`（现有硬校验）② GT wire 像素永不进负监督 ③ target 逐像素与原物料化一致。

### PREREG 修正单（2026-10-08，Lead 裁决，结果前合规）— 误伤率门槛的对象与单位

**裁决内容**（依据实测证据，见下）：
1. **门槛绑定对象改为 NEG（唯一训练监督来源）**：立法目的是"训练监督不得把真导线标为负"。
   NEG 误伤率 = **0.00%（构造性）**：NEG ∩ dilate(GT_wire,3) = ∅，物料化三项硬断言把关。**门槛绑定 NEG → 达标 ✓**
2. **单位取笔画级**："多少真导线被标注"的正确单位是笔画条数，不是像素：
   22 089 条被标注笔画中 **328 条（1.49% < 2% ✓）** 为真导线笔画；像素级 4.9% 是被 328 条长笔画（总线/回路）
   按长度加权放大（纯几何交叉色带交叠仅 1 986px）。
3. **停止旋钮**：0020 参数收紧实验（companion 40→8 / span 0.8→0.95 / rect 500→2000 / interior 5→10）
   对 0241 有效（8.5%→0）但对 0020 **反而恶化（44%→55%）**——误伤源是少数长导线在结构上长得像装饰
   （interior 抓大器件框内走线、canvas_span/rect 抓长总线/回路），非可参数化问题，继续拧旋钮=过拟合，停止。
4. **不关 interior 签名**：符号内部笔画分层的价值真实存在，且它不进监督。
5. SIG 仅做分层与规则迁移核验，**不进监督**；训练监督只吃 NEG。

**条件（写入 REPORT）**：a) 本修正单列一节（含笔画级/像素级加权分析 + 0020 不可改进证据）；
b) 效力门槛 **G-w10 残余 FP 桶覆盖率 ≥60% 不变**（G-w1 实测主目标）；c) 物料化三项硬断言保持。

## 2. 负样本注入方式（三选一，预注册）：**显式负 patch**

**机制（单一选择，一次训练只改这一件事）**：

- **A 监督掩码翻转（负 patch 的像素语义）**：在 `NEG_eff` 内，清除 ignore（组件 bbox/文本 bbox 的 ignore 让位于负监督），
  target 强制为 0 → 这些像素第一次获得"不是 wire"的显式监督。NEG_eff 之外逐像素保持现行语义（对照干净）。
- **B 显式负 patch（采样密度）**：每样本按 NEG 密度 top-K 裁剪**非网格偏移 tile**（512×512，钳制在画布内），
  target/ignore 按 A 规则生成，record_id 前缀 `negp_`。作用：难负样本（符号/装饰笔画密集区）在每个 epoch 被重复采样，
  同时偏移裁剪本身构成平移增广。

**为什么选它（放弃另两项的理由）**：
- *loss 加权*：现有 `compute_wire_loss` 只有二值 valid 掩码，加权需改共享训练代码（pin 线程正在训练，风险不可控）；
  且装饰笔画已是监督负样本，缺的是**采样密度与监督缺口**，不是梯度权重。留作唯一一次"loss 曲线定向调整"的备选。
- *两阶段（笔画级分类）*：需新模型任务 + 推理管线集成 + 模型锁新条目，超出本轮 scope 与 GPU 预算，列入后续计划。
- 显式负 patch **纯数据层实现、零共享代码改动**，lineage 可诚实记录（materializer config 扩展字段含 NEG 标注文件 sha256；
  run_document 由内容寻址 run_id 覆盖），失败零成本回退（candidate-only，生产锁不动）。

### 物料化与绑定（诚实 lineage）

- 新物化脚本 `work/opt90-wire/opt90_materialize.py`：以 `WireJunctionMaterializeConfig` 扩展字段
  （`negative_label_path`/`negative_label_sha256`/`neg_dilate_px`/`protect_margin_px`/`neg_patch_per_sample`/`neg_patch_offset_seed`）
  走**原 `materialize_wire_junction` 发布协议**（内容寻址 `runs/<run_id>` + `latest.json` + summary 哈希），
  输入仍是 `data/derived/dataset300-v2/wire_paths.jsonl` 等四份冻结清单（`derived_config_hash` 不变 → preflight lineage 不变）。
- 物化输出根：`work/opt90-wire/materialized/`，训练时以环境变量
  `PCB_WIRE_JUNCTION_MATERIALIZED_ROOT` 指向（torch_backend 原生支持的 override，不动 `artifacts/training/...` 固定根）。
- 训练配置：`configs/train/opt90_wire_v1_negpatch.json`（只新增）＝ `dataset300_v2_wire_v5_fresh_allgate.json` 的 wire 任务
  （tiny_wire_unet_v1 / 512 / bs1×accum16 / lr2e-4 / 20ep / fp16 / skeleton_f1≥0.55）**逐字段不变**，仅换物化根 → 单变量对照。

## 3. 训练与验证流程

1. 负标注器跑 train240（240 样本）→ 误伤率量化 → 达标才进下一步；
2. 物料化（新 run）→ 自检：target∩ignore=∅、NEG_eff 像素全部 valid 且 target=0、原网格 tile 与基线逐像素一致（除 NEG 翻转区）；
3. `preflight_cli` → `--smoke`（先 CPU smoke；GPU smoke 与正式训练均需 Lead 批 GPU1 槽位，pin 先我后）；
4. 正式训练（串行，不扫网格；最多一次 loss 曲线定向调整）→ candidate bundle（不写锁）；
5. 绑定候选锁（仿 `work/module-opt80-20261007/r07/candidate-lock-ppocr.json`，新条目诚实 `candidate-evaluation`，生产锁不动），
   run_r07 模式跑 val30 + `tools/evaluate_candidate.py`，对照 r12 终局出 `work/opt90-wire/REPORT.md`。

## 4. 预注册验收门槛（结果前冻结，val30 / qna-v1 本地口径）

| # | 指标 | 门槛 | 级别 |
|---|---|---|---|
| G-w1 | wire precision | **≥ 0.90** | 主目标（未达如实封口） |
| G-w2 | wire recall | ≥ 0.85 硬下限（期望 ≥ 0.90） | 硬门槛 |
| G-w3 | wire F1 | ≥ 0.7687 | 不降 |
| G-w4 | graph pair | TP ≥ 24 且 FP ≤ 8（graph 精度目标 ≥ 0.85） | 不回退 |
| G-w5 | S5 conditional | ≥ 0.8352 | 不降 |
| G-w6 | designator / value / pin.number e2e | ≥ 0.585 / 0.504 / 0.158 | 不降（G3） |
| G-w7 | 字节级确定 30/30（triple + disk rebuild） | 全 true | G4 |
| G-w8 | comp F1 / pin F1 | ≥ 0.6648 / ≥ 0.2913 | 随行不降 |
| G-w9 | 989+ 单测 | 零新增失败 | 回归 |
| G-w10 | **残余 FP 桶覆盖率**（机制证据） | r12 归因"完全孤立·非媒介·无彩色符号笔画"桶 FP px 抑制率 **≥ 60%**（新模型 vs r12 终局同口径重放） | 主目标的机制证明（监督对象真的被学到） |

G-w10 测量口径：val30 上用 r12b/r12e 同口径分层（OCR 文本框内 / 器件 bbox 边线带 / pin 邻域 / 完全孤立）+ 二级分解
（贴 GT 线 / 8px 内 / 孤立），取"完全孤立"中非媒介线段的**无彩色符号笔画桶**像素质量，基线 = r12-integration 终局
（残余 FP 51 131 px），抑制率 = 1 − 新模型同桶 px / 基线同桶 px。
**基线实测（2026-10-08，r12_bucket_coverage.py 重放 r12-integration 终局 predictions，逐数复现 r12 报告
pred 165 522 / FP 51 131）：桶 px = 24 722**（isolated 40 043 / 非文本非边带 33 004 / 其中无彩色 24 722、彩色 8 282）。
→ G-w10 达标线：新模型同桶 **≤ 9 889 px**（抑制率 ≥60%）。
**s6fix 口径基线（Lead 确认的验收口径）**：run-s6fix predictions 同脚本测得桶 px = **24 529**（FP 50 867），
→ 达标线 **≤ 9 812 px**（0.4 × 24 529）。两口径都报告，以 s6fix 口径为验收判定。

附：负标注误伤率 < 2%（保护前）；注入后 target∩ignore=∅ 恒成立。

## 5. 风险与回退

- 若 G-w1 未达但 G-w2~G-w9 全过：如实报告 FP 残余构成（对照 §1 NEG 覆盖率），判定是否仍 model-bound；
- 若 recall 跌破 0.85：保护带/翻转区审计 → 唯一一次定向调整（备选：NEG 区 BCE 负权重下调 或 保护带 3→5px）；
- 任何门槛失败都不进晋级通道；candidate-only，`models/models.lock.json` 不动；
- 纪律：禁下载、禁伪造 lineage、禁碰他人 scope（`work/opt90-{pin,ocr,lead}`、`src/association/`）、GPU1 串行未批不启动。

## 6. 证据路径约定

- 负标注：`work/opt90-wire/neg-labels/neg_stats.json`（method 段含全部口径哈希与复现命令；逐样本误伤率）、
  `NEG_<sample>.png`（训练负区域，与 GT 保护性不相交）、`SIG_<sample>.png`（四签名标注，保护前，误伤率门禁对象）；
- G-w10 桶覆盖：`work/opt90-wire/r12_bucket_coverage.py` + `r12_bucket_coverage.json`（val30，r12b/r12e 同口径）；
- 物化：`work/opt90-wire/materialized/runs/<run_id>/summary.json`（lineage 含 NEG 文件哈希）；
- 训练：`work/opt90-wire/train-v1/`（training_log.jsonl / metrics / bundle）；
- 验证：`work/opt90-wire/eval-val30/`（run_r07 模式 + evaluate_candidate 输出）；
- 总报告：`work/opt90-wire/REPORT.md`。
