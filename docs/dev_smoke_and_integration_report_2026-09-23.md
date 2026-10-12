# 开发图冒烟与集成报告（2026-09-23）

对应批次目标：按 `docs/初步构建复核与下一步优化建议_2026-09-22.md` 第一批 5 项目标完成集成，
先跑跨模块测试，再跑固定开发图全链路冒烟，产出与当前源码绑定的证据。
**本报告全部数字为本轮真实运行所得，不复用 2026-09-14 或任何历史结果。**

## 0. 结论摘要

| 项 | 结论 | implemented | fixture_passed | development_e2e | release_ready |
|---|---|---|---|---|---|
| 1 A01 pin 解析契约统一 | `expand_official_pin_numbers("pin_")` → `()`，审计器与 serializer 共用同一解析实现 | ✓ | ✓ | — | — |
| 2 质量入口接入生产链路 | `serialize_official_with_quality` 进 orchestrator/submission，五类主状态分类 | ✓ | ✓ | ✓ | — |
| 3 A04 逐对象真实漏斗 | 三类对象一致漏斗 + 逐对象 ID/原因 + not_measured 语义 | ✓ | ✓ | ✓ | — |
| 4 cap 真实传递 + A08 | 可见边长预算（floor 12 / pitch 8px / cap 128）贯通至 decoder；真实密集 IC 截断证据 | ✓ | ✓ | ✓ | — |
| 5 virtual_repairs/breaks 入 IR | 补线来源与坐标进 IR 与最终 JSON，serializer 显式 edges 策略 | ✓ | ✓ | — | — |

`release_ready` 全部为否：无官方 evaluator、无 GPU 时延/P95、无 Docker 演练、无独立确认。

## 1. 源码与模型绑定

| 绑定项 | 值 |
|---|---|
| git HEAD | `3510d644122510b11775f4c834e738faaca30287` |
| 未提交改动 | 75 个文件（本轮工作区改动，未提交） |
| 源码清单 | 262 个 `src/`+`scripts/`+`configs/`+`data_tools/` 文件 |
| source_sha256 | `0fdfc9ac1ce0a722c72124e6c587e6dd205960404c4809735936ea94e2395d8d` |
| models.lock.json SHA-256 | `af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413`（未改动） |
| 模型版本 | component-34d79a6d… / ocr-e4212287… / pin-ec3c50dc… / wire-3235b9e0…（readiness 逐文件 SHA 验证通过） |

机器可读汇总：`artifacts/audits/dev_smoke_20260923_summary.json`；运行产物：`work/dev-smoke-20260923/`。

## 2. 逐项：命令、结果、限制、失败样本、后续依赖

### 任务1 A01 pin 解析契约统一

- 命令：`python -m unittest tests.unit.test_contract_audit_a01`；`python -m unittest discover -s tests -p "test_*.py"`（`.venv-test`）
- 结果：`test_comma_multi_pin_numbers_expand` 由失败转通过；全套 709 tests OK（skipped=73）。
  `contract_audit.py` 与 `official_schema.py` 共用 `EMPTY_PIN_NUMBER_TOKENS` / `PIN_NUMBER_SPLIT_RE` /
  `split_official_pin_numbers`；`"pin_"` 空后缀、`pin_None`/`pin_null` 噪声值均返回 `()`。
- 失败样本（修复前）：`expand_official_pin_numbers("pin_") == ("pin_",)`（错误）→ 现 `()`。
- 限制：210 目标全量契约清点本轮未重跑（属 A00/A01 交付，本批仅修边界）。
- 后续依赖：契约审计全量复跑并入 integration snapshot。
- 证据等级：implemented + fixture_passed。

### 任务2 质量入口接入真实编排与提交

- 命令：
  - `python -m unittest tests.unit.test_submission_agent`（含合成 provider 跨层 5 状态测试，走真实 `infer_one`）
  - `python -m pcbparse.submission_agent -input work/dev-smoke-20260923/bad-inputs -output work/dev-smoke-20260923/bad-out`（`.venv-train`，真实生产 runtime）
- 结果：`serialize_official_with_quality` 已是编排序列化唯一入口（`InferenceResult.quality` 随结果下发）；
  submission `decision_log.json` 每 case 携带 quality 与五类主状态 `strict_success / degraded_nonempty /
  empty_result / fallback_empty / failed_to_write`，空结果与异常回退不再混记。
  真实运行：2 个坏输入 → `empty_result`（blank，legitimate empty）+ `fallback_empty`（corrupt，`ImageLoadError`，失败原因保留）。
- 失败样本：`corrupt.png` → `fallback_empty`，error=`ImageLoadError: cannot decode image …`；
  `blank.png` → `empty_result`（error=None，区分于异常回退）。
- 限制：开发冒烟 5 图均为 `degraded_nonempty`，本轮未观察到 `strict_success` 实例（缺 key 在 degraded 下被裁剪）。
- 后续依赖：30 张 val 批量跑出五状态分布。
- 证据等级：implemented + fixture_passed + development_e2e。

### 任务3 A04 逐对象真实漏斗

- 命令：
  - `python -m unittest tests.unit.test_error_funnel_a04`
  - `python scripts/error_funnel_a04.py --prediction-manifest work/dev-smoke-20260923/a04-manifest.jsonl --dataset-root Dataset300/Dataset300 --official-output-dir work/dev-smoke-20260923/preds-by-case --output-dir work/dev-smoke-20260923/a04`
- 结果：`error_funnel.json` 为三类对象一致漏斗（文本字段 / Pin / Net），同一分母、逐阶段
  included/excluded 对象 ID 与原因；未测阶段为 `not_measured` + `count=null`（不写 0）。
  预测清单绑定 `image_sha256`/`source_sha256`/`config_sha256`/`model_lock_sha256`，GT 仅在评估器内使用。
  oracle 分解：crossing 组合因无可信 crossing GT 标记 `uncomputable`（`oracle_decomposition.json`）。

  5 张开发图累计漏斗（count / denominator=853 文本字段、902 pin、270 net）：

  | 阶段 | 文本字段 | Pin | Net |
  |---|---|---|---|
  | 检出/定位/匹配 | 115 / 853 | 129 / 902 | 6 / 270 |
  | 字符正确 / owner 正确 | 50 / 853 | 124 / 902 | 0 / 270 |
  | 字段正确 / 编号名称正确 | 50 / 853 | 5 / 902 | 0 / 270 |
  | owner 正确 / 完整命中 | 28 / 853 | 2 / 902 | — |
  | 输出正确 | 17 / 853 | — | — |

- 失败样本（首失败阶段）：文本 738 个停在 `detected`（如 `0001:C_1.designator`）、
  Pin 773 个停在 `localized`（如 `0001:pin:P_1`）、Net 264 个停在 `matched`（如 `0001:net:2`）。
  完整样本清单：`work/dev-smoke-20260923/failure_samples.json`。
- 限制：30 张 val 漏斗未跑；crossing oracle 不可测；raw-v1 匹配规则≠官方 evaluator。
- 后续依赖：30 张 val 全量漏斗 + 错误 ROI 导出（A04 验收后半）。
- 证据等级：implemented + fixture_passed + development_e2e。

### 任务4 cap=12 真实传递验证 + A08 真实测试

- 命令：
  - `python -m unittest tests.unit.test_pin_geometry_a08`（`.venv-train`，16 项全过；3 项 torch 门控在 `.venv-test` skip）
  - `python -m pcbparse.cli infer --input work/dev-smoke-20260923/inputs/0087 --output work/dev-smoke-20260923/preds-fixed12 --config work/dev-smoke-20260923/config-fixed12.yaml …`
- 结果：
  - 链路验证：`configs/pipeline.edarace.yaml` pin 段 → `_parse_pin_inference_overrides` →
    `TorchPinProvider.config` → `decode_pin_heatmaps(max_detections=…)`；真实运行时配置快照两次实证
    （budget 模式与 fixed-12 模式各自的 `pin_inference_overrides` 均出现在 readiness 输出中）。
  - 可见边长预算：`clamp(ceil(周长/8px), 12, 128)` 逐组件传入 decoder；16/32/64 合成热图验证
    fixed-12 截断 top-12 且 4px/5px 匹配可解释，budget 模式全保留、FP=0。
  - 真实密集 IC（0087，含 Dataset300 最大 143-pin Box）cap 前后对比
    （`work/dev-smoke-20260923/dense_ic_0087_cap_comparison.json`）：

    | 口径 | budget 模式 | fixed cap=12 |
    |---|---|---|
    | 最密组件保留 pin 数 | 17 | 12（截断 5） |
    | 全图保留 pin 总数 | 50 | 45 |
    | PinF1 @4px / @5px | 0.0138 / 0.0138 | 0.0140 / 0.0140 |
    | 编号 exact | 0 / 235 | 0 / 235 |

- 失败样本：0087 上两种预算的 PinF1 均 ≈0.014、编号 exact 0/235——
  截断修复不解决检测召回与编号关联（见任务3 Pin 漏斗）；fixed-12 对最密组件的 5-pin 截断为真实发生。
- 限制：全图集 top-k 截断率需 GPU 扫描（not_measured）；编号 exact / 完整 PinF1 依赖 OCR 编号关联修复后再评。
- 后续依赖：GPU 复跑 top-k 截断率；编号关联修复后重评 4px/5px PinF1。
- 证据等级：implemented + fixture_passed + development_e2e。

### 任务5 virtual_repairs / breaks 来源与坐标入 IR

- 命令：`python -m unittest tests.integration.test_virtual_repair_lineage`；`python -m unittest discover -s tests -p "test_*.py"`
- 结果：`VirtualGapRepair` 携带 bridge 像素与映射坐标、新增像素数；受影响 polyline/segment 挂
  `Evidence(source="repair", virtual=True, solid_wire=False)` 且 `WireSegment.synthetic` 置位；
  `CircuitIR.repairs`（`rule_id=mask_closing_virtual_gap`，object_ids=受影响 segment）与
  `metadata["wire_breaks"]`（含恢复到原图的 left/right 点）贯通；serializer 显式声明官方 edges 来源策略
  （`wire_provenance` 质量块：relation-only 补线永不成为 edges）。小 mask 夹具经
  CompositePerceptionProvider 追踪到 IR dump JSON 与最终官方 JSON，连接标记不丢失；
  邻近平行线（8px 间距）/断口跨线/文字噪声夹具确认无大规模误并网。
- 失败样本：真实开发图上 `wire_gap_closing_radius_px=1` 产生大量补线记录（0001: 13406、0016: 39157），
  已全部可追溯（此前会静默并线）；后续需评估该阈值是否引发误并。
- 限制：补线证据尚未回流调节 `wire_gap_closing_radius_px` 取值。
- 后续依赖：用 A04 net 漏斗回归评估 gap closing 阈值。
- 证据等级：implemented + fixture_passed。

## 3. 固定开发图冒烟（development_e2e）

- 命令链（`.venv-train`，CPU）：
  1. `python -m pcbparse.cli healthcheck --require-ready --config configs/pipeline.edarace.yaml --models-lock models/models.lock.json` → ready=true，4 模型逐文件 SHA 验证
  2. `python -m pcbparse.cli infer --input work/dev-smoke-20260923/inputs --output work/dev-smoke-20260923/preds --config configs/pipeline.edarace.yaml --profile competition --provider production --models-lock models/models.lock.json --evaluation-ir-dir work/dev-smoke-20260923/ir` → **succeeded 5 / failed 0**
  3. `python -m pcbparse.evaluation.raw_cli --prediction-manifest … --output work/dev-smoke-20260923/raw_v1_metrics.json`
  4. `python scripts/error_funnel_a04.py …`（见任务3）
  5. `python -m pcbparse.submission_agent -input work/dev-smoke-20260923/bad-inputs -output work/dev-smoke-20260923/bad-out`（见任务2）

固定输入（覆盖目标 → 实际选样）：

| case | 覆盖点 | 质量分类（真实回放） | 说明 |
|---|---|---|---|
| 0087-KiCad | 密集 IC（143-pin Box，全集最大） | degraded_nonempty | 裁剪 24 组件 / 42 pin |
| 0088-Datasheet | 重复位号（R2、C2、R3 各重复） | degraded_nonempty | 裁剪 13 / 34 |
| 0016-Datasheet | 高边密度（153 边，跨线富集） | degraded_nonempty | 裁剪 38 / 95 |
| 0001-KiCad | 复杂连线（133 边、72-pin IC） | degraded_nonempty | 裁剪 83 / 156 |
| 0005-kicad | 小图基线 | degraded_nonempty | 裁剪 11 / 14 |
| blank.png | 空输入 | empty_result | error=None，合法空输入单独标记 |
| corrupt.png | 坏输入 | fallback_empty | ImageLoadError，失败原因保留 |

每张成功图均有：可回放 IR（`work/dev-smoke-20260923/ir/*.circuit-ir.json`）、真实质量分类
（`work/dev-smoke-20260923/quality_replay.json`，含 pin 自动编号数、裁剪组件/pin 数、补线数）、
非空指标（raw-v1 + A04 漏斗）；失败输入有失败原因（decision_log）。

raw-v1 masked 汇总（5 图，本地评估器，非官方口径）：

| 指标 | P | R | F1 |
|---|---|---|---|
| component 检测 | 0.837 | 0.122 | 0.214 |
| pin 检测 | 0.620 | 0.137 | 0.225 |
| wire 中线 | 0.366 | 0.884 | 0.518 |
| graph 成对连通 | 0.029 | 0.009 | 0.013 |

## 4. 限制（不得越界宣称）

1. ONNX CUDA 回退 CPU（`cublasLt64_13.dll` 缺失，日志有据）——无 GPU 时延/P95/显存证据。
2. raw-v1 是本地 masked 评估器，不是官方 evaluator；上述指标不得当作竞赛成绩。
3. 5 张开发图是 development_e2e 证据，不是泛化证据；公开 210 图不是新 holdout。
4. 无 Docker 冒烟、冷启动、三次确定性复跑、独立确认——release_ready 全线为否。
5. 0087 上 cap 前后 PinF1 均 ≈0.014：截断策略修复与检测召回/编号关联是不同问题，后者是当前主要损失
   （编号 exact 0/235、编号/名称正确仅 5/902）。
6. Net 漏斗 membership 正确 0/270：net 层与 GT 完全未对齐，图层问题未在本批修复范围内。

## 5. 后续依赖

1. 获取官方 evaluator 后才能谈 release_ready；竞赛口径指标一律以官方为准。
2. 30 张 val 全量 A04 漏斗 + 错误 ROI 导出（含 4px/5px 双口径）。
3. GPU 环境修复（或官方镜像）后补：top-k 截断率全图集统计、时延/P95/显存、三次确定性。
4. OCR pin 编号关联修复（当前编号 exact≈0 的主因），然后重评 PinF1。
5. Net membership 层对齐（graph 构建 vs GT connectivity），再评估 `wire_gap_closing_radius_px` 阈值。
6. Docker 冒烟/离线启动/坏图处理/连续批量，绑定冻结候选后做发布验收。
