# REPORT — opt90-wire：wire 分割模型笔画监督重训（goal 线 B）

日期：2026-10-08 · 线程：wire-model-tune · 状态：**训练前证据已闭环，等待 GPU 槽**
预注册文件：`work/opt90-wire/DESIGN.md`（含 2026-10-08 PREREG 修正单，Lead 裁决）

---

## 1. 结论先行

（训练与验收结果完成后填写；本节只在数据侧先记一句）
**数据侧已完成对"监督缺口"的定量证明与修复**：现行 wire 物料化的 ignore 语义使 **1 579 236 px 的符号/装饰笔画
（住在组件/文本 bbox ignore 区）从未被要求输出负样本**（唯一像素口径；tile 级累计 3 204 315 px 含负 patch 重叠复计）
——这正是 r12 判定 model-bound 的残余 FP 的同分布人口。本次把它翻转为显式负监督（显式负 patch 机制），
target 逐像素不变、GT wire 构造性零误伤。

## 2. 方法与预注册（详见 DESIGN.md）

- **训练负样本 NEG（唯一监督来源）**：train240 上**生产 wire 模型**（models.lock wire entry
  `3235b9e083190e8d…`）按生产推理路径（tile_image_view 生产 tiling → TorchWireAdapter.segment →
  生产 `_vectorize_masks`）输出的**误检像素**（距 GT wire 中心线 >3px，tolerance 与 masked_metrics qna-v1 一致）。
- **四签名标注 SIG（分层/核验工具，不进监督）**：生产 `filter_decorative_wire_segments` 跑在同一套模型笔画
  segment 上（GT 组件 bbox/文本点 far-gate），canvas_span / companion / rect_outline / interior 分层。
- **注入机制（三选一预注册 = 显式负 patch）**：
  A 监督掩码翻转：NEG_eff = NEG − dilate(GT,3) 内清 ignore、target 强制 0 → 显式负监督；
  B 显式负 tile：每样本按 NEG 密度 top-K 非网格偏移 512 tile（record `negp_*`）提高难负采样密度。
- 两阶段（笔画级分类）列入后续计划；loss 加权是唯一一次"loss 曲线定向调整"的备选（未动共享代码）。

## 3. 负标注器迁移与误伤率量化

**口径**（可复现命令与全部哈希记录于 `neg-labels/neg_stats*.json.method`）：
`PYTHONPATH=src <runtime python> work/opt90-wire/opt90_neg_labeler.py --workers 48`

| 量 | 数值 | 说明 |
|---|---|---|
| NEG（训练负区域）误伤率 | **0.00%** | NEG ∩ dilate(GT_wire,3) = ∅（构造性 + 物料化逐 tile 断言） |
| SIG 像素级误伤率 | 5.48%（262 457 / 4 788 220 px） | 带宽指标（SIG dilate 2px 色带 ∩ GT±3px） |
| SIG 笔画级误伤率 | **1.48%（360 / 24 252 条）** | "真导线被标注"的正确单位（>50% 像素压在 GT 上的笔画） |
| 纯几何交叉交叠 | 1 932 px | 装饰笔画色带与导线交叉点，非标注错误 |

**参数收紧实验（预注册程序执行并留证）**：companion 40→8 / span 0.8→0.95 / rect 500→2000 / interior 5→10：
0241 修好（8.5%→0.0%），**0020 反而恶化（44.4%→54.7%）**——误伤源是少数长导线在结构上长得像装饰
（interior 抓大器件框内走线、canvas_span/rect 抓长总线/回路），非可参数化问题。停止拧旋钮（继续=过拟合）。
证据：`neg-labels/neg_stats*.json`、`sig_stroke_analysis.json`（237/240 样本覆盖，3 张超密图 SIG 后台补跑）、
`neg_tight_probe.log`、叠加图 `overlays/overlay_0020.png`（蓝=SIG 装饰框/符号框，绿=GT wire，红=NEG）。

**PREREG 修正（Lead 裁决，结果前）**：门槛绑定对象改为 NEG（立法目的"训练监督不得把真导线标为负"），
单位取笔画级；SIG 不进监督。详见 DESIGN.md 修正单。

## 4. 显式负 patch 物料化

- run_id `a9d08faecb6281993aa1f7d04f7389ece11b6bdbaeca94993a98188e6319beae`（内容寻址，原发布协议）；
  materializer_config_sha256 `52bb89bddf5ecf3032e8494a2b07ba8285c38f0816ee5b3d7d3797068f666260`；
  NEG 树哈希 `32c4f51fff369f1fcf55feae71e2e8b558edf4728065f1c88c3550ea28163b50`（240 NEG png 全部入 asset 清单）。
- **规模**：wire_tile_count 2695（全 split）；训练集 **1458 tile = 1023 网格 + 435 显式负 patch**；val 154。
  NEG 唯一像素 **3 047 609 px**（train240 全画布）；其中**新翻转的显式负监督 = 1 579 236 px**（原 ignore 区，
  网格 tile 唯一口径；负 patch 复计 1 625 079 px，tile 级累计 3 204 315 px 如实注明为重叠计数）；
  另 118 354 px 本已是监督背景（采样加权覆盖）。148 样本有注入；target 像素总数 1 469 700（逐像素不变）。
- **硬断言（三项，逐 tile）**：① target∩ignore=∅（训练 loader 亦强校验）② GT wire 像素永不进负监督
  ③ target 逐像素与原物料化语义一致。val/test 零注入；不可用稀疏样本全程保持 ignore（不泄漏监督）。
- selfcheck 全过（走训练 loader 全量 1721 条记录校验）：`selfcheck_full.log`。
- 线路（诚实 lineage）：输入=四份冻结清单（derived_config_hash 3381e705…不变），materializer config 扩展字段
  含 NEG 树哈希，run_document 内容寻址覆盖。

## 5. 训练

- 配置 `configs/train/opt90_wire_v1_negpatch.json`：与基线 `dataset300_v2_wire_v5_fresh_allgate.json`
  **逐字段一致、仅 name 不同**（单变量对照，diff 证据已存）；tiny_wire_unet_v1 / 512 / bs1×accum16 / lr2e-4 / 20ep。
- preflight 过（lineage 哈希全对）；CPU 训练 smoke 全链路过（checkpoint / model.pt / 导出）。
- 物化根绑定：`PCB_WIRE_JUNCTION_MATERIALIZED_ROOT=work/opt90-wire/materialized`（原生 override，固定根不动）。
- **正式训练：（结果后填写：时长、逐 epoch 表、最终 val 指标、bundle 路径）**

## 6. 验收（门槛 = DESIGN §4 冻结值）

（结果后填写表格；对照基线 r12-integration 终局：wire P 0.6911 / R 0.8659 / F1 0.7687 / graph 25→8；
Lead 确认口径：F1≥0.7687@s6fix、graph TP≥22/FP≤8、G-w10 桶抑制 ≥60%（≤9 889px）、S5≥0.8352、字段 e2e 不降、G4）
G-w10 基线（已钉死）：r12 终局口径桶 px **24 722**；**s6fix 口径（验收判定）桶 px 24 529 → 达标线 ≤ 9 812px**；
重放脚本 `r12_bucket_coverage.py` 逐数复现 r12 报告（pred 165 522 / FP 51 131），s6fix 基线 `r12_bucket_s6fix_baseline.json`。

## 7. 证据路径清单

| 内容 | 路径 |
|---|---|
| 预注册 + PREREG 修正单 | `work/opt90-wire/DESIGN.md` |
| 负标注统计（含口径哈希/复现命令） | `work/opt90-wire/neg-labels/neg_stats*.json` |
| NEG/SIG 掩码 | `work/opt90-wire/neg-labels/NEG_*.png`、`SIG_*.png` |
| 笔画级误伤归因 | `work/opt90-wire/sig_stroke_analysis.json` |
| 叠加取证 | `work/opt90-wire/overlays/overlay_0020.png` 等 |
| 物料化 receipt / run | `work/opt90-wire/materialize_receipt.json`、`materialized/runs/a9d08fae…` |
| selfcheck | `work/opt90-wire/selfcheck_full.log` |
| G-w10 基线 | `work/opt90-wire/r12_bucket_baseline.json` |
| 训练 | `work/opt90-wire/train-v1/`（结果后） |
| 验收 | `work/opt90-wire/eval-val30/`（结果后） |

## 8. 纪律与限制

- 全程零 src/ 改动、零 models/models.lock.json 改动、零他人 scope 侵入；训练 GPU1 串行（经 Lead 排队批准）。
- 本地口径声明：所有指标为 qna-v1 本地代理口径，非竞赛成绩；官方 evaluator 未获得；val30 非独立 holdout。
- 5 张超密图（0003/0041/0046/0050/0076）NEG 走 mask-only 路径（NEG 定义只依赖模型掩码+GT，与向量化无关，
  口径一致）；其 SIG 分层标注为后台补跑项，不进监督、不影响任何门槛。
