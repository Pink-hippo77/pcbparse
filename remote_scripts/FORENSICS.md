# FORENSICS.md — 组件检测 FN 分层归因（goal 线 D · work/opt90-comp）

日期：2026-10-08。写域：`work/opt90-comp/`。**GT-ACCESS: diagnostic_only** —— 全部 GT 只进本目录下
明确标记的离线诊断脚本，不进任何推理/训练输入。基线栈：`work/opt90-lead/run-s6fix`（val30，
config 3142f31b、PP-OCR 候选锁、blessed 组件/pin/wire）。检测匹配口径 =
`pcbparse.evaluation.masked_metrics._component_metrics`（IoU≥0.5、类别无关、最大基数匹配）。

## 0. 结论（先说答案，含一处对任务前提的证伪）

**"46 例 S7 失败中 40 例=真主人组件未被检出、component 召回 0.554 是检测器召回"这两个前提不成立。**
检测器在感知阶段已经检出了那些组件；它们死于输出前的 **key 裁剪**（`_prune_unrepresentable`，
orchestrator.py:82-118：无非空 designator key 的组件一律删除，官方 schema 以位号为 key、禁止伪造字段）。
val30 全量：感知阶段 **1153** 个组件 → **438 个因无 key 被删** → 715（= 616 TP + 99 FP）。
检测器裁剪前实测 **P 0.8005 / R 0.7541**（14 例复刻口径），已高于流水线召回的**结构性上限 0.6349**。

| 分叉问题 | 证据 | 判定 |
|---|---|---|
| FN 主体是几何错位（近失）？ | IR 层 near_miss 仅 64/496 = 12.9% | 否 |
| FN 主体是完全未检出？ | IR 层 no_overlap 422/496 = 85.1%，**但这 422 里 141+ 例在裁剪前有 IoU≥0.5 的检测框** | 表象是、实质否 |
| **FN 主体是什么？** | **裁剪前检测框覆盖 58% 的 FN（141/243，14 例口径）；全量 438 个组件被 key 裁剪** | **key 绑定 + 降级裁剪** |
| 是小目标/尺度问题？ | 召回按尺寸平坦（0.508–0.606） | 否 |
| 是图纸域问题？ | "灾难图"实为 key 裁剪重灾区（0234 删 50、0280 删 42）；唯一真塌陷图 0092（分数标定） | 域假象为主 |
| 检测器还能怎么帮？ | 框质量→绑定几何；裁剪前 P 0.80 有 FP 上升空间 | 辅助，非主杠杆 |

## 1. 关键机制链（自底向上）

```
感知(512/64 切图 + 0.3 分数门 + 类别NMS 0.45 + cap128) ──► 1153 个组件
   │  （0280: 86 过门 → NMS 后 50；0092: 1419 raw → 9 过门）
   ▼
association: resolve_component_keys（OCR 位号文本→组件绑定）
   │  绑上 key 的留下；V/Gnd 无 key 者由 _assign_qna_special_coordinate_keys 发 '∅'（Q&A 坐标口径）
   ▼
_prune_unrepresentable（degraded 模式）──► 无 key 组件全删（438/1153 = 38%）
   ▼
最终 IR 715 组件（TP 616 / FP 99 / FN 496，recall 0.5540）
```

证据：
- 复刻生产组件阶段（`replicate_component_phase.py`，逐行同款调用链/配置/模型）：0280 → **50** 个组件
  （C 29、R 9、Gnd 8、L 2、Buzzer 2），run-s6fix IR 只剩 **8 个 Gnd**；0092 → 9，IR 剩 2。
- 全量裁剪账本（`prune-stats.json`，读 IR 自记录的 `metadata.degraded_submission_prune`）：

| 案例 | 感知检出 | 因无key被裁 | 剩余 | | 案例 | 感知检出 | 被裁 | 剩余 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 0234 | 60 | **50** | 10 | | 0171 | 127 | 24 | 103 |
| 0280 | 50 | **42** | 8 | | 0202 | 84 | 16 | 68 |
| 0238 | 54 | **28** | 26 | | 0140 | 85 | 15 | 70 |
| 0089 | 27 | **24** | 3 | | 0190 | 84 | 11 | 73 |
| 0091 | 33 | **28** | 5 | | 0144 | 43 | 19 | 24 |
| 0283 | 27 | **24** | 3 | | 0092 | 22 | 20 | 2 |
| **合计** | **1153** | **438** | **715** | | | | | |

- 规则出处：`src/pcbparse/pipeline/orchestrator.py:82-118`（`valid_components = {key 非空}`）与
  `src/pcbparse/association/text_component.py:63-95`（仅 V/Gnd 发 '∅' key）。**本线禁改这两个文件。**

## 2. 检测器真实水平（裁剪前，preprune-metrics.json，14 例复刻）

| 口径 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 裁剪前（感知输出） | 325 | 81 | 106 | **0.8005** | **0.7541** | 0.7766 |
| 裁剪后（流水线 IR，全量30例基线） | 616 | 99 | 496 | 0.8615 | 0.5540 | 0.6743 |

FN 重分类（243 个 FN，14 例）：**141（58%）落在裁剪前检测框 IoU≥0.5 窗口内**（检到了、被裁掉）、
22 近失（0.1–0.5）、80 真盲。S7 的 40 个 owner 同理（部分已验证：0037 C_6 近失 0.445 属几何，
其余待全量 raw 覆盖后入 preprune-metrics.json 的 s7_owners_preprune）。

**唯一真·检测塌陷图 = 0092**：raw 1419 → 仅 9 个过 0.3 门（分数普遍 0.02–0.15，类别/几何正确，
如 C_2 raw512 IoU 0.625 分数 0.0675 类别 R）；对照好图 0140 raw 226、126 个 ≥0.5。属**分数标定塌陷**
（风格域敏感），另有 0089/0283（同图）等 5 图在 key 裁剪上受灾。raw 候选全量落盘
`raw-candidates.jsonl` / `quick-raw-*.jsonl`（score≥0.02，512/64 主切图 + 256/32 回退尺度）。

## 3. 算术上限封证（schema-ceiling.json）

官方 schema 以 designator 为组件 key；Q&A 只豁免 V/Gnd（'∅' 坐标匹配）。val30 GT 1112 个组件：

| GT 类别 | 数量 | 可表示性 |
|---|---:|---|
| V/Gnd（无位号，走 '∅' 规则） | 342 | ✅ 恒可表示 |
| 非 V/Gnd 且标注有位号 | 364 | ✅ 可表示（需绑定成功） |
| **非 V/Gnd 且无位号标注** | **406** | ❌ **schema 结构性不可表示** |

→ **流水线 component 召回硬上限 = 706/1112 = 0.6349**（当前 0.5540，理论最大 +0.081）。
且检测器裁剪前召回 0.7541 已超过该上限——**检测侧已无兑换空间，剩余增量全部取决于
"364 个有位号组件的文本绑定成功率"**（association 域）。

## 4. 裁剪前的其他分层事实（保留作检测侧改进的依据）

- 框几何：TP 群体系统内缩（dw p50 −1.27px / dh p50 −1.67px、面积比 p50 0.91；36 个 TP 贴 0.5 线）；
  确定性框重放（`box-replay.json`）最好档 margin+1px 仅 +10 TP；S7 owner 翻正上限 8/40
  （`s7-box-replay.json`）→ 框定标是弱杠杆，只可作绑定几何的辅助。
- 尺寸：召回按尺寸平坦（<20px 0.508 / 20–40px 0.566 / 40–80px 0.601 / 80–160px 0.606 / ≥160px 0.360），
  非小目标问题；S7 的 40 个 owner 均 <40px（0190 的 12 例是 26×10px 堆叠条形电阻阵列，
  `probe/crop-0190-ladder.png`）。
- 类别：IR 层 NET_input/PIN/Buzzer/M/AMP/Net_*/Dc 全灭（98 GT）——但其主因是这些类别的 GT 多无位号
  （结构性不可表示）而非模型盲；裁剪前 raw 中可见其候选（如 0092 PIN 0.337、0280 Buzzer 1.0）。
- 训练底座欠训练：v5-fullmix 自身 val bbox_map50 0.2501（贴 0.25 门槛线）、recall50 0.3165。

## 5. 对两条目标的结论

| 目标 | 归因结论 | 检测器可为 | 上限 |
|---|---|---|---|
| S7 0.8345→0.90 | 40 例 = 位号文本绑错/真主人无 key 被裁（绑定域） | 框质量间接助绑定 ≤+8/40 | 需 association 侧修复绑定 |
| component 召回 ↑ | 438 裁剪 + 406 结构性不可表示 | 裁剪前 R 0.754→↑ 无流水线兑换 | **0.6349 硬上限** |
| component 精度 ≥0.8615 | 裁剪本身是精度机制（裁剪前 P 0.80 < 裁剪后 0.86） | 降 FP 提升裁剪前 P | 训练可为但非门槛所需 |

**按预注册纪律（"不过即如实报告"）：本线若无 Lead 另行授权，不开训练战役**——训练无法兑换
两个目标指标；裁决请求已发 Lead（选项 A 封证收线 / B 改口径训练 / C 配置包，见往来消息）。

## 6. 证据索引

| 文件 | 内容 |
|---|---|
| `replicate_component_phase.py` | 逐行复刻生产组件阶段（0280: 50、0092: 9 的决定性证据） |
| `prune_stats.py` → `prune-stats.json` | 30 例裁剪账本（1153→715，逐例 dropped_ids） |
| `preprune_metrics.py` → `preprune-metrics.json` | 裁剪前 P/R/F1、FN 重分类、S7 owner 覆盖 |
| `schema_ceiling.py` → `schema-ceiling.json` | 406 结构性不可表示、上限 0.6349 |
| `fn_attribution.py` → `fn-attribution.json` | 496 FN IR 层分层 + S7 联接 + FP 配对 |
| `fn_strata_recall.py` → `fn-strata.json` | 类别/尺寸基率召回 + S7 40 明细 |
| `group_strata.py` → `group-strata.json` | 灾难图 vs 其余类别召回（域假象定位） |
| `nearmiss_geometry.py` → `nearmiss-geometry.json` | TP/近失框边差、IoU 直方图 |
| `box_replay.py` / `s7_box_replay.py` | 框定标确定性重放（弱杠杆封证） |
| `raw_detector_probe.py` / `quick_probe.py` → `raw-candidates.jsonl`/`quick-raw-*.jsonl` | 模型原始输出（≥0.02） |
| `raw_analysis.py` / `gate_replay.py` | 分数分布、类别混淆、门槛重放 |
| `dataset_stats.py` / `case_profile.py` | 训练数据普查、图像域画像 |
| `dump_ir.py` / `dump_meta.py` / `dump_meta2.py` | IR 组件/元数据解剖（裁剪证据） |
| `probe/*.png` | 目视证据（FN 叠加、raw 叠加、预览） |
