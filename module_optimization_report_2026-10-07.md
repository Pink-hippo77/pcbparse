# P0/P1 优化执行报告（2026-10-07，服务器 llm）

基线：**B14-current-server**（source_set `b1f4a0ab…`，lock `af35e129…`，生效配置 `ef17dbc…` = `configs/pipeline.edarace.yaml`）。
评估口径：**qna-v1 本地代理**（`raw_cli`，非官方 evaluator）；val30 为开发回归集，非独立 holdout；官方成绩一律 null。
执行环境：仅服务器运行/评估/写入；GPU1 单卡租约 + UUID 校验；推理进程 GT/10GT 读取审计阻断（`sidecar_access=false`）。

---

## 0. 结论速览

| 项 | 状态 | 结果 |
|---|---|---|
| P0-A pinNumber 规则冲突 | **validated** | 冲突裁决：撤销无原文支撑的"官方转换器自动编号"契约声明，标 `unresolved`；预测侧维持"剪枝/拒绝、绝不自动编号"；serializer 公共入口新增 ERROR 诊断自检（R02）；50/50 定向测试通过，全量单测零新增失败 |
| P0-B owner/key 可序列化率 | **implemented（账本）+ 机制在位/无收益** | 漏斗账本确立根因链：**OCR 识别失败 260/364 → 器件无 key → 其 pin 不可序列化**；前缀策略机制已在源码（cycle84 验证 val30 零变差）；联合竞争（85 例 pinnumber 误路由）预注册待 R05 窗口 |
| P0-C 生产 readiness | **validated** | `healthcheck --require-ready` 由 unready → **ready**（`configs/pipeline.yaml` 修复为锁内 Torch OCR 绑定；Paddle 命名候选归档为 unready）；4/4 绑定锁定测试通过 |
| P1-A component 召回 | **rejected（有 Pareto 证据）** | Gnd/V 第二尺度恢复单因素：召回 +2.58%（TP +11/FN -11）但精度 -6.43%（FP +37）、pin F1 -1.11%，四门槛 G1/G2 未过→拒绝；恢复通道有效但需候选分数门槛（预注册第二次单因素） |
| P1-B pin 归属绑定 | **pre-registered** | owner-unique + 单调行唯一化机制已预注册（含 cycle91 负对照门槛），待 R05/R06 窗口实现 |
| P1-C wire 过检 | **rejected（有证据）** | 度-2 转角链式重建 **-12.3% F1、TP 52→37，双门槛失败→拒绝**；共线合并 +1.18% 未达采纳线；FP 根因重新归因为**过检笔画**而非碎线粒度 |

不变式全数保持：Validator 0 ERROR 才序列化、无占位造号、无 sidecar、三顶层键、模型锁未动、字节确定性未动。

---

## 1. P0-A：pinNumber 规则冲突裁决与合同冻结（R01+R02）

### 1.1 冲突与裁决

`contract_audit.py` 曾声明 `pin_number_policy.empty_allowed_auto_number=true, status=confirmed`
（"空 pinNumber 允许、官方转换器自动编号"）。核对结论：

- 一手《赛题历史 Q&A》原文**不在服务器**；二手核对记录（`work/module-relative10-20261005-v1/qna-applicability-audit-20261005.md`）
  唯一相关条款是：**GT 侧** VCC/GND 特殊编号随机生成、序号不计分，且明确"**系统不得自行伪造编号绕过 Validator**"。
- AGENTS 硬边界禁止自动编号/占位填充/伪造字段。

**裁决**：撤销该声明，`official_converter_blank_number_behavior = unresolved`；预测侧保持
`prune_upstream_or_reject`。**不开放自动编号通道**——先前分析中"655 个缺号 pin 可经官方自动编号找回"的
预期不成立，其真正修复路径是 OCR/绑定覆盖率（P0-B/R03/R05），不是合同放宽。
（注：VCC/GND 坐标身份例外维持不动：`∅` key + 坐标匹配 + `key_score=0`，有 rule 证据，见
`tests/unit/test_qna_special_coordinate_identity.py`。）

### 1.2 实现与锁定

- `docs/qna_contract_table_2026-10-07.md`：14 条 Q&A 契约对照表（来源/实现/状态），#7 冲突裁决全文。
- `src/pcbparse/evaluation/contract_audit.py`：`pin_number_policy` 改为
  `{multi_number_split_on_comma: true(confirmed), empty_or_placeholder_numbers: prune_upstream_or_reject,
  auto_numbering_at_prediction: forbidden, official_converter_blank_number_behavior: unresolved, status: unresolved}`。
- `src/pcbparse/output/official_serializer.py`（R02）：`serialize_official` / `serialize_official_with_quality`
  公共入口新增 `_reject_error_diagnostics`——携带 `severity=ERROR` 诊断的 IR（含 `state=VALIDATED` 的矛盾 IR）
  直接拒绝；WARNING/INFO 不拦截；无 ERROR 合法 IR 输出逐字节不变。
  缺 key/缺号拒绝、无原地改写、稳定排序、虚拟关系不生成实线等既有行为全部保留。
- 新增 `tests/unit/test_serializer_boundary_a03.py`（10 项）：矛盾 IR 探针由接受→拒绝、干净 IR 字节稳定、
  缺 key/缺号拒绝保留（`""/␣␣/∅/None/null/Ø` 全部拒绝）、永不自动编号、契约策略锁定。

### 1.3 验证

- 定向：`test_serializer_boundary_a03` + `test_validation_serializer` + `test_contract_audit_a01` +
  `test_qna_special_coordinate_identity` = **50/50 通过**。
- 全量 unit：834 项，1 fail + 4 errors + 12 skip——与基线失败集合**完全一致**（组件 v5 preflight 文件、
  `onnx` 模块、Faster R-CNN 预训练权重、OCR v5 warm-start 文件四类环境资产缺失），**零新增失败**。
- 该四类资产按 R02 应"恢复真实资产或提供可复现替代 fixture、不得伪造 lineage、不得新增 skip 掩盖"：
  资产本体不在服务器且禁止下载，**延期**（见 §7）。

---

## 2. P0-B：owner/key 可序列化率——账本、根因与机制状态（R05）

### 2.1 逐阶段账本（B14，val30，qna-v1）

设计器文本漏斗（364 个合格位号文本框，`ocr-designator-funnel`）：

| 首败阶段 | 数量 | 占比 |
|---|---:|---:|
| text_recognized（OCR 未读对字符串） | **260** | 71.4% |
| box_detected（文本框未检出） | 26 | 7.1% |
| field_classified（角色分类错） | 15 | 4.1% |
| owner_bound（归属绑定失败） | 9 | 2.5% |
| 全部通过 | 54 | 14.8% |

失败样本的角色误路由：**pinnumber 85**、designator 18、pinname 17、空 20、value 2。
多视图审计：364 合格 → 任意视图读到精确 token 90 → 加角色约束 73 → 加 owner 约束 63（可恢复样本 27）。

**根因链修正**：先前"1,897 个 pin 死于 owner key 门禁 → 主因绑定层"的判断需要修正为——
pin 漏斗的 key 损失是**设计器覆盖率**的下游投影，而设计器覆盖率的主损失在 **OCR 字符串识别**（260/364）。
owner 绑定本身仅 9 例首败。因此 P0-B 的度量改进空间不在 key 校验规则，而在 OCR 识别与角色竞争（R03/R05/R11）。

### 2.2 机制状态

- 前缀白名单/角色约束（`_is_valid_designator` + `_prefix_compatible` + `_canonical_designator`）**已在生产源码**；
  fit214 训练推导的 `prefix-policy.json`（47 个允许前缀）经 cycle84 预关联守卫验证 **val30 changed_cases=0**
  （内层 74 观测 73 正确、加权精度 0.9925 与基线一致）。结论：该机制无 val30 指标收益，作为防误键（LM334 型号当 key）的稳健性护栏保留。
- 真正的机制实验（85 例 pinnumber 误路由的联合角色竞争）**预注册**至 R05 窗口（`work/module-opt-20261007/r06/run-001/prereg-p1b-pin-binding.md` 同批门槛）。

### 2.3 基线度量固化

`work/module-opt-20261007/eval_qna_v1.py` 以统一口径重算 B14：comp F1 **0.5301**、pin F1 **0.2125**、
graph pairwise F1 **0.01434**、wire F1 **0.5496**（此前散见于不同 run 的口径混用问题由该脚本消除）。

---

## 3. P0-C：生产 readiness 恢复（R03）

**问题**：`healthcheck --require-ready --config configs/pipeline.yaml` 失败——
`ocr.recognizer_backend: "en_PP-OCRv5_mobile_rec"` 是 **PaddleOCR 模型名**（`recognizer_backend.py` 纯字符串→Paddle 构造），
而服务器三个运行时（edarace-py312 / ocr-paddle33-cu118-py312 / test-extra）**均无 `paddleocr`**、`models/` 无 Paddle 权重、
模型锁不含该资产。该配置同时是 `scripts/audit_release_hard_gates.py` 等 4 个发布脚本的默认生产配置——发布路径被阻断。

**修复（授权 Torch OCR 绑定替代）**：

1. `configs/pipeline.yaml` 的 `ocr` 段改绑 **models.lock 的 blessed Torch OCR bundle**（与 B14 `pipeline.edarace.yaml`
   生效绑定一致），`recognition_candidate_count` 5→0（多候选/beam 是未采纳负实验，R04 记录 FP 23→26）。
2. 原 Paddle 候选完整归档于 `work/module-opt-20261007/r03/paddle-recognizer-candidate.yaml`（candidate-unready，
   注明缺模块/缺权重；除非按 R03 用真实资产重建，不得冒充生产绑定）。
3. 新增 `tests/unit/test_production_ocr_binding.py`（4 项）锁定：生产配置只允许锁内资产绑定、beam 保持关闭、候选归档不丢失。

**验证**：`healthcheck --require-ready` 于 `configs/pipeline.yaml` 与 `configs/pipeline.edarace.yaml` 双双
**`status: ready, problems: []`**，四任务 lineage verified，runtime smoke ok。

---

## 4. P1-A：component 检测（小符号专项召回，单因素）——**rejected（有 Pareto 证据）**

**机制**：`image.component_targeted_recovery_enabled=true` + `categories=["Gnd","V"]`
（composite_provider 注释明确该开关面向"符号在主 tile 中过小的类别"的第二尺度 256/32 恢复）。
唯一差异即此开关组（生效配置 `a86fc8ba…` vs B14 `ef17dbc…`）；其余与 B14 完全一致。

**运行**：GPU1 租约 + UUID/空闲守卫、30 输入 SHA 校验、10GT/非输入读取审计阻断；
**30/30 成功、0 失败**（含 0202 长尾图，单图均未破 600s），整批 2321.7s < 7000s；锁 SHA 全程校验不变。

**qna-v1 配对结果（val30）**：

| 指标 | B14 | 候选 | 绝对Δ | 相对Δ |
|---|---:|---:|---:|---:|
| component F1 | 0.5301 | 0.5280 | -0.0021 | -0.39% |
| component precision | 0.8557 | 0.8007 | -0.0550 | -6.43% |
| component recall | 0.3840 | 0.3939 | **+0.0099** | **+2.58%** |
| pin F1 | 0.2125 | 0.2102 | -0.0024 | -1.11% |
| graph pairwise F1 | 0.014337 | 0.014345 | +0.000009 | +0.06% |
| wire F1 | 0.5496 | 0.5502 | +0.0006 | +0.11% |

检测计数：TP 427→**438（+11）**、FN 685→**674（-11）**、FP 72→**109（+37）**。

**验收（预注册四门槛）**：G1 comp F1 不降 ✗（-0.39%）、G2 pin F1 不降 ✗（-1.11%）、G3 graph ✗→✓、G4 wire ✓
→ **all_gates_pass=false → rejected**。

**解读与下一步**：机制确有真实召回效应（找回 11 个漏检符号，方向正确），但恢复通道同时放入 37 个误检，
精度损失吞掉收益并外溢到 pin。第二因素候选（仅对恢复通道加分数门槛/每类上限）未在本轮扫描（stop_rule）；
按 R07 窗口作为预注册第二次单因素执行。证据：`work/module-opt-20261007/r07/run-001/`
（`preflight.json`、`baseline-freeze.json`、`p1a-qna-v1-metrics.json`、`acceptance.json`）。

---

## 5. P1-B：pin 归属绑定（预注册，延期至 R05/R06 窗口）

证据：val30 可见精确编号 283 → 绑定失败 102 → **绑错 59**（cycle89）。cycle90/91 行对齐修复是负对照
（编号 TP 150→199 但连接 TP 79→73、FP 364→421）：单字段修复会破坏拓扑。

预注册机制：**owner-unique + 单调行唯一化**（Hungarian 后确定性修复通道，仅重分配已观测 token，禁发明，
保留 UNKNOWN）；感知冻结 A/B；验收五门槛（旧正确字段零丢失、净提升 ≥20 目标/≥98% 预筛、
**全量 graph TP≥24 且 FP≤64**、其余指标不退化、strict 30/30 + 3 次字节确定）。全文见
`work/module-opt-20261007/r06/run-001/prereg-p1b-pin-binding.md`。

本轮未启动实现的原因：该机制属于 R05"联合角色/owner/pin 竞争"的子集，实现与验证是完整研究量级；
仓促上线正落入 cycle91 已证伪的"单字段赢、拓扑输"陷阱。

---

## 6. P1-C：wire 过检治理（R08，实验已跑，机制被证据拒绝）

背景：wire 像素 F1 0.5496（P 低 R 高）；端点代理显示预测线段数约为 GT 6 倍。两个假设：
碎线粒度（分段过碎）或过检笔画（非导线被识别为导线）。

**实验（`work/module-opt-20261007/r08/run-001/`）**：在 B14 官方输出上做输出级、GT-free 的
**度-2 顶点链式重建**（允许转角；pin/junction 保护点断链；T 接天然断链；不产生零长边），
对照（a）精确共线合并（既有 qa-netline-coalesce 方案）与（b）B14 原始；端点代理 5px 最大基数一对一匹配。

| 变体 | TP | FP | FN | F1 | 相对 B14 |
|---|---:|---:|---:|---:|---:|
| B14 原始 | 52 | 23095 | 3846 | 0.003845 | — |
| 精确共线合并 | 52 | 22779 | 3846 | 0.003891 | +1.18% |
| **度-2 转角链式（候选）** | **37** | 18009 | 3861 | 0.003372 | **-12.31%** |

- 链式候选按预注册双门槛判定：**TP 52→37 下降 → 拒绝（rejected）**（per-case 8 升/15 平/7 降）。
- 共线合并 TP 不变、+1.18% 相对，低于采纳阈值，不写入生产。
- **归因修正**：合并减少了 FP 却丢 TP，说明 GT 折线端点并不能靠"碎线拼接"重建——
  wire 的主导误差是**过检笔画**（文字/器件内笔画/错误坐标进入线段集），与阈值实验（v17 降阈值像素↑网络↓）
  一致。后续 R08 应按四层对应做笔画归因与分类抑制，而非继续合并粒度。
- 不变量：变换不动 components/pins/网络成员（逐 case 断言 CIR 恒等）；生产源码未改。

---

## 7. 延期项与理由（未做的优化）

| 项 | 状态 | 理由 |
|---|---|---|
| R02 环境资产恢复（组件 v5 preflight、onnx、Faster R-CNN 权重、OCR v5 warm-start） | deferred | 资产不在服务器且禁止下载/伪造 lineage；可在独立导出环境补 `onnx` 后跑导出验证（待授权） |
| R04 OCR crop/trace 契约 | deferred | 依赖 R03 OCR 资产盘点 |
| R05 联合角色/owner/pin 竞争（含 85 误路由） | pre-registered | 研究量级实现；门槛已冻结 |
| R06 主体 ROI→pin 因果实验 | deferred | 需 GPU1 窗口排队；v7 历史目录问题需先修 |
| R09/R10 图误接与标签并网 | deferred | 依赖 R08 笔画归因结论 |
| R11 训练 | deferred | 本轮禁止自动启动训练；三轮 medium 微调已退化，需先审计监督语义 |
| R12 运行时 | deferred | 0202 长尾需分阶段计时先行；并发需先隔离共享诊断状态 |
| R13 静态库 | deferred | 等最终 cpntlibrary |

## 8. 回归与不变式状态

- 全量 unit：834 项，失败集合与基线一致（环境资产类），零新增失败；A03/A03-binding 新增 14 项全绿。
- `healthcheck --require-ready`：双生产配置 **ready**。
- 模型锁未改（`af35e129…` 全程校验）；无新模型/下载；GT 仅进离线评估器；`sidecar_access=false` 全程审计。
- 源码变更集（before→after SHA-256 记录于 `work/module-opt-20261007/delivery-manifest-20261007.json`）：
  `official_serializer.py`、`contract_audit.py`、`configs/pipeline.yaml` + 新增 2 测试 + 1 契约文档。
  原字节备份在 `work/module-opt-20261007/source-before/`（含 SHA 校验的 `pipeline.yaml` 原件重建）。
- R14 组合验收（3 次字节确定性、error_manifest、输出审计）**尚未执行**——P1-A 结果并入后进行。

## 9. 复现命令

```bash
PY=/home/data/yinjx/pcbparse-competition/runtime/edarace-py312/bin/python
cd /home/data/yinjx/pcbparse-competition/project
PYTHONPATH=src $PY -m unittest tests.unit.test_serializer_boundary_a03 tests.unit.test_production_ocr_binding -v
PYTHONPATH=src $PY -m pcbparse.cli healthcheck --require-ready --config configs/pipeline.yaml --models-lock models/models.lock.json
$PY work/module-opt-20261007/eval_qna_v1.py work/module-relative10-20261006-v14/current-prod-config-val30-v1/evaluation/predictions.jsonl work/module-opt-20261007/r05/run-001/b14-eval-manifest.jsonl work/module-opt-20261007/r05/run-001/b14-qna-v1-metrics.json
$PY work/module-opt-20261007/r08/run-001/run_netline_chain_v1.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $PY work/module-opt-20261007/r07/run-001/run_p1a_targeted_recovery.py
```

## 10. 声明

所有分数为 qna-v1/raw-v1 本地代理开发口径，**不是竞赛成绩**（官方 evaluator 未获得）；val30 为反复使用的开发
回归集，不构成独立泛化证据；95% 四项目标是目标而非当前结果。
