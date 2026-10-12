# REPORT.md — opt90-comp：组件检测器微调线（goal 线 D）· 负结果封证

日期：2026-10-08。写域：`work/opt90-comp/`、`configs/train/opt90_comp_*`（本线未新增训练配置）。
**GT-ACCESS: diagnostic_only**（GT 只进本目录离线诊断脚本，零 GT 进推理/训练）。GPU 消耗：**0**。
生产锁 `models/models.lock.json` 全程只读；共享物化/派生路径零写入；`src/` 零改动。

## 0. 结论

**训练战役被第一性原理归因否决，按预注册纪律"不过即如实报告"封证收线（Lead 裁决选项 A）。**

任务前提"S7 的 40 例=真主人组件未被检出、component 召回 0.554 是检测器召回"被证伪：
**检测器在感知阶段已经检出了这些组件，它们死于输出前的 key 裁剪**（`_prune_unrepresentable`，
orchestrator.py:82-118：无非空位号 key 的组件一律删除；官方 schema 以位号为组件 key，禁止伪造字段）。
val30 全量账本：感知 **1153** 个组件 → **438 个因无 key 被裁** → 715（= 616 TP + 99 FP）。
**核心悖论：检测器裁剪前召回 0.754 > 流水线召回的结构性上限 0.6349**——继续训检测器在流水线指标上
没有兑换空间。S7 与召回的真根源在 key 绑定漏斗（association 域，Lead 主攻），已完整移交。

## 1. 五条证据链（自底向上，全部零 GPU 离线复算）

1. **复刻生产组件阶段**（`replicate_component_phase.py`：同模型/同配置/逐行同款调用链）：
   0280 产出 **50** 个组件（C 29、R 9、Gnd 8、L 2、Buzzer 2；含 C 0.9993、R 0.9981 等高分框），
   run-s6fix 的 IR 只剩 **8 个 Gnd**；0092 产出 9（R5/Box/PIN/C/Gnd），IR 剩 2。**丢失在感知之后**。
   留档：`replicate-0280.txt`、`replicate-0092.txt`。
2. **丢失点定位**：`orchestrator._prune_unrepresentable` 只保留 `key` 非空的组件；
   唯一豁免是 `text_component._assign_qna_special_coordinate_keys`（text_component.py:63-95）
   给 **V/Gnd 发 '∅' key**（Q&A 坐标匹配口径）——所以 0280 只剩 Gnd、其余全灭。
3. **全量裁剪账本**（`prune-stats.json`，IR 自记录 `metadata.degraded_submission_prune`）：
   30 例合计 1153 → 715（**裁 438 = 38%**）。重灾图：0234 60→10（裁 50）、0280 50→8（裁 42）、
   0238 54→26、0091 33→5、0089 27→3、0283 27→3；即使好图也裁（0171 127→103、0144 43→24）。
4. **检测器真实水平**（`preprune-metrics.json`，14 例复刻口径）：裁剪前 **P 0.8005 / R 0.7541 / F1 0.7766**
   （裁剪后 0.8615 / 0.5540）。FN 重分类（243 个）：**141（58%）落在裁剪前检测框 IoU≥0.5 窗口内**、
   22 近失、80 真盲。→ "完全未检出"是表象。
5. **算术上限**（`schema-ceiling.json`）：1112 个 GT 中 V/Gnd 342（'∅' 规则恒可表示）、非 V/Gnd 有
   标注位号 364（可表示、需绑定成功）、**非 V/Gnd 无标注位号 406**（按位号口径不可表示）→
   **召回上限 = 706/1112 = 0.6349**（当前 0.5540，检测侧理论最大 +0.081 且全靠绑定兑现）。

## 2. 分层归因备忘（详见 FORENSICS.md）

- IR 层 FN 496 = no_overlap 422 / near_miss 64 / weak 10；近失翻正的确定性重放上限仅 +10 TP
  （`box-replay.json`，margin+1px 最好档），S7 owner 翻正 ≤8/40（`s7-box-replay.json`）→ 框定标弱杠杆。
- 框几何有系统内缩（TP 面积比 p50 0.91、36 个 TP 贴 0.5 线），可作绑定几何的辅助项，非主杠杆。
- 尺寸分层平坦（0.508–0.606），非小目标问题；0190 的 12 例 S7 owner 是 26×10px 堆叠条形电阻
  （`probe/crop-0190-ladder.png`）。
- 唯一真·检测塌陷 = 0092（raw 1419 → 仅 9 过 0.3 门，分数 0.02–0.15、类别/几何正确），
  对照好图 0140（raw 226、126 个 ≥0.5）→ 分数标定塌陷（风格域敏感），仅此一图属模型侧。

## 3. 负结果判定表（预注册门槛）

| 门槛（任务书） | 判定 | 原因 |
|---|---|---|
| S7 0.8345→0.90 | **不测（封证）** | 40/46 失败=位号文本绑错/真主人无 key 被裁，属绑定域；检测侧几何上限 ≤+8/40 |
| component 召回 ↑ | **不测（封证）** | 上限 0.6349；检测器裁剪前 0.754 已超上限，兑换在绑定 |
| component 精度 ≥0.8615 不降 | 不适用 | 未开训练，基线保持 |
| S5 / 四字段 / pin / graph / G4 | 不适用 | 未动任何模型/配置，基线保持 |

**是否达到"检测侧算术上限封证"替代条款**：是——§1.5 的 0.6349 上限 + 悖论（检测器 0.754 > 上限）
共同构成封证：检测侧改进对两个目标指标的边际兑换为零，训练战役按纪律不开。

## 4. 移交（详见 HANDOVER.md）

1. **438 个被裁组件逐例清单**（`HANDOVER-pruned-components.{json,md}`）：含分数/类别/框（212 例已重建）、
   GT 匹配与 **GT 位号标注有无**列（179 例匹配 GT：153 无标注位号 / 26 有）。
2. **0280/0092 裁剪前后对照**（`replicate-0280.txt`/`replicate-0092.txt` + 本报告 §1.1）。
3. **复跑工具** `replicate_component_phase.py`（任意案例一键复刻感知输出，零 GT）。
4. **GT 治理项（供 GT_GOVERNANCE.md 第 5 项）**：406/1112 GT 组件无标注位号（类别分布：
   NET_input 58/58、LED 18/18、Box 58/90、C 91/228、R 82/217、Switch 26/35、PIN 18/21、L 14/21 等）；
   口径注记：其中 C/R/Box 部分在图上**可能存在未标注但可见的位号文本**（如 0280 的 "C35"），
   若绑定兑现上限可上浮——该开放量受 S5/S6/S7 文本漏斗约束，已在 HANDOVER 标注。

## 5. 纪律记录

- 零 GPU（全部 CPU 离线诊断）；训练战役未开；未新增训练配置；未产出 candidate bundle。
- `models/models.lock.json`、`artifacts/training/materialized/`、他人 write scope（work/opt90-{pin,wire,ocr,lead}、
  `src/pcbparse/association/`、`src/pcbparse/training/torch_backend.py`）零改动。
- 训练用隔离工作区已就绪备用（`work/opt90-comp/ws/` + `materialized-default/`，binding
  a406ca0ef815…，counts 2885/12789 与 blessed 物化一致）——若 Lead 未来重启训练战役可直接使用。
- 一切结论的证据路径见 §6；GT 只进 `gt_access: diagnostic_only` 脚本。

## 6. 证据索引

| 文件 | 内容 |
|---|---|
| `FORENSICS.md` | 分层归因定稿（五条证据链展开） |
| `HANDOVER.md` / `HANDOVER-pruned-components.{json,md}` | 移交清单 |
| `replicate_component_phase.py` / `replicate-0280.txt` / `replicate-0092.txt` | 感知阶段复刻（裁剪前输出） |
| `prune-stats.json` | 30 例裁剪账本（1153→715） |
| `preprune-metrics.json` | 裁剪前 P/R/F1、FN 重分类、S7 owner 覆盖 |
| `schema-ceiling.json` | 406/1112 不可表示、上限 0.6349、类别分布 |
| `fn-attribution.json` / `fn-strata.json` / `group-strata.json` | FN 分层（IR 层） |
| `nearmiss-geometry.json` / `box-replay.json` / `s7-box-replay.json` | 框几何与定标重放 |
| `raw-candidates.jsonl` / `quick-raw-*.jsonl` / `raw-probe.json` | 模型原始输出（score≥0.02，双尺度） |
| `dataset-stats.json` / `case-profile.json` | 训练数据普查、图像域画像 |
| `probe/*.png` | 目视证据 |
