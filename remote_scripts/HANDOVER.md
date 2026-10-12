# HANDOVER.md — opt90-comp 移交包（→ association 线 / GT_GOVERNANCE）

日期：2026-10-08。写域：`work/opt90-comp/`。GT-ACCESS: diagnostic_only。
背景：本线（组件检测器微调）经归因证伪任务前提后按 Lead 裁决封证（选项 A）；
以下为给 key 绑定漏斗（association 主攻）与 GT 治理清单的移交物。

## 1. 438 个被 key 裁剪的组件逐例清单

- `HANDOVER-pruned-components.json`（结构化，438 条）/ `HANDOVER-pruned-components.md`（可读表）。
- 字段：case、internal_id（= `degraded_submission_prune.component_ids_dropped` 原样）、
  score / category / bbox（212 条已由确定性复刻重建；226 条 box_unavailable=无 raw 转储的案例，
  id 仍全量保留）、best_gt（IoU≥0.5 匹配的 GT）、gt_category、gt_designator、
  **gt_designator_annotated**（GT 治理列：True=标注有位号 / False=标注无位号 / None=无 GT 匹配或未重建框）。
- 已核验统计（212 条重建框内）：**179 条匹配 GT 真组件**（其中 **153 条 GT 无标注位号**、26 条有）、
  33 条无 GT 匹配（本身是 FP 候选，被裁剪反而护了精度）。
- 复算工具：`handover_pruned.py`（随 raw 转储覆盖扩大可重跑补齐 226 条未重建框）。

## 2. 0280 / 0092 裁剪前后对照（决定性证据）

| 案例 | 感知阶段（`replicate_component_phase.py` 复刻） | 最终 IR（run-s6fix） | 差额 |
|---|---|---|---|
| 0280 | **50** 个组件：C 29 / R 9 / Gnd 8 / L 2 / Buzzer 2（最高分 C 0.9997、Buzzer 1.0、R 0.9981） | **8** 个（全 Gnd，key='∅'） | **42 个无 key 被裁** |
| 0092 | **9** 个：R 5 / Box 1 / PIN 1 / C 1 / Gnd 1（1419 个 raw 候选、仅 9 过 0.3 分数门） | **2** 个 | 7 个被裁 + 检测侧分数塌陷 |

逐行留档：`replicate-0280.txt`、`replicate-0092.txt`。
复跑任意案例（零 GT、纯感知）：
```bash
/home/data/yinjx/pcbparse-competition/runtime/ocr-paddle33-cu118-py312/bin/python \
  work/opt90-comp/replicate_component_phase.py <case>
```

## 3. 裁剪机制定位（供 association 侧修复参考）

- 规则：`src/pcbparse/pipeline/orchestrator.py:82-118` `_prune_unrepresentable`
  （`valid_components = {key 非空}`；`metadata.degraded_submission_prune` 自记账本）。
- 唯一豁免：`src/pcbparse/association/text_component.py:63-95`
  `_assign_qna_special_coordinate_keys` 给 **V/Gnd 无 key 者发 '∅'**（Q&A 坐标匹配口径）。
- 因此输出侧存活条件：V/Gnd 类 ∨ 有绑定成功的位号文本（`resolve_component_keys` +
  `_is_valid_designator`）。**绑定漏斗每漏一个位号，就整箱丢失一个真组件**（438/1153 = 38%）。
- 全量账本：`prune-stats.json`（30 例 dropped_ids / kept / detected 全字段）。

## 4. GT 治理项（→ GT_GOVERNANCE.md 第 5 项，用户侧裁决项）

**406/1112 = 36.5% 的 GT 组件在标注中无位号，其中非 V/Gnd 者按"位号为 key"的官方 schema
不可表示**（V/Gnd 342 例另有 '∅' 豁免）。类别分布（total / 无标注位号）：

| 类别 | 无标注/总数 | | 类别 | 无标注/总数 |
|---|---|---|---|---|
| NET_input | 58/58 | | Box | 58/90 |
| LED | 18/18 | | C | 91/228 |
| Buzzer | 9/9 | | R | 82/217 |
| Switch | 26/35 | | PIN | 18/21 |
| AMP/ANT/Battary/M/Other/Net_*/Dc/Fuse/Jumper | 全部无标注 | | L | 14/21 |

口径注记（诚实边界）：0.6349 = 706/1112 是**按"位号只认标注/可绑定位号文本"的口径上限**。
其中 C/R/Box 部分在图上可能存在**未标注但可见**的位号文本（例：0280 图面有 "C35/C36" 而标注缺），
若绑定兑现则上限可上浮；该开放量受 S5/S6/S7 文本漏斗约束（S5 0.9176 × S6 0.862 × S7 0.8345），
且现状 438 个被裁组件中无一兑现。已在移交中显式标注，供治理清单裁决"补标注 vs 承认不可表示"。

## 5. 检测器侧残留事实（若未来重启检测训练，直接可用）

- 隔离训练工作区已就绪：`work/opt90-comp/ws/`（硬链接 Dataset300/data/models + 物化软链）；
  物化 `work/opt90-comp/materialized-default/component-pin`（默认配置，binding a406ca0ef815…，
  counts 2885/12789 与 blessed 一致，可复现）。
- 检测器裁剪前 P 0.8005 / R 0.7541（14 例）：**P 是弱侧**（FP 81，含重复框）；
  框系统内缩（面积比 p50 0.91、36 TP 贴 0.5 线）可辅助绑定几何；0092 类分数标定塌陷
  （风格域）是唯一模型侧真缺陷；训练底座 v5-fullmix 欠训练（val mAP50 0.2501 贴门槛线）。
- raw 候选转储（`raw-candidates.jsonl` + `quick-raw-*.jsonl`，score≥0.02 双尺度）保留可复用。
