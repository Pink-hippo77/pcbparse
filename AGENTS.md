# AGENTS.md

PCB 原理图自动解析（赛题六）。以 **Circuit-IR** 为唯一内部表示，分层流水线：
`perception`（模型适配）→ `association`（OCR/Pin 语义绑定）→ `graph`（Physical Graph）
→ `nets`（Electrical Net）→ `validation`（严格 Validator）→ `output`（官方 JSON 序列化）。

## 命令

所有命令需要 `$env:PYTHONPATH=(Resolve-Path '.\src').Path`（CI 用 `pip install --no-deps -e .` 替代）。
`data_tools` 会自举注入 src，但测试与文档约定仍是 `PYTHONPATH=src`。

```powershell
# 健康检查：exit 0 只代表进程存活
python -m pcbparse.cli healthcheck
# 生产就绪：真实加载 adapter + 固定图 smoke；与模型锁/配置绑定
python -m pcbparse.cli healthcheck --require-ready --config configs/pipeline.yaml --models-lock models/models.lock.json

# 全量测试（正式口径是 unittest，不是 pytest；unit 先、integration 后）
.\scripts\run_tests.ps1 -Python <python>
# 等价 CI 命令
python -m unittest discover -s tests -p "test_*.py" -v
# 单个测试
python -m unittest tests.unit.test_validation_serializer -v
# 或（.venv-test 可用 pytest）
python -m pytest tests/unit/test_validation_serializer.py -q

# 数据底座（顺序：ingest → audit → split → derive）
python -m data_tools.pcbnet_ingest --dataset-root .\Dataset300\Dataset300 --output .\data\manifests\dataset300.jsonl --expected-count 300
python -m data_tools.audit --dataset-root .\Dataset300\Dataset300 --output .\artifacts\audits\dataset300_audit.json   # --strict 在原始数据上预期失败（19 warning）
python -m data_tools.split_grouped --manifest .\data\manifests\dataset300.jsonl --output .\data\splits\dataset300_split.jsonl --seed 20260812
python -m data_tools.derive_datasets --src .\Dataset300\Dataset300 --out .\data\derived\dataset300-v2 --version dataset300-v2 --split-manifest .\data\splits\dataset300_split.jsonl

# 训练：先 --smoke 再正式；preflight 校验 lineage
python -m pcbparse.training.preflight_cli --config .\configs\train\<cfg>.json --workspace-root .
python -m pcbparse.training.torch_train_cli --config .\configs\train\<cfg>.json --task pin --output .\work\<run> --workspace-root . --device cuda --precision fp16 --smoke
# 晋级模型锁（唯一途径；需完整 bundle + 证据报告）
python -m data_tools.model_promotion --bundle <4 个 bundle.manifest.json> --junction-mode deterministic_vectorizer --promote --output <新锁> --report-output <报告>
```

## 环境

- 三个 venv 职责不同，不要互换：`.venv-train`（torch/训练）、`.venv-test`（pytest 单测）、`.venv-yolo`（ultralytics）。
- 运行时依赖仅 NumPy/Pillow/Pydantic（Python 3.11/3.12）。
- `requirements.edarace.txt` 明确**不安装 torch**：官方 Docker 镜像拥有 CUDA 匹配的 PyTorch。
- 本机已知故障：ONNX CUDA 缺 `cublasLt64_13.dll` → CPU fallback，时延不可当 GPU P95 证据。
- 训练串行（单 GPU）：`scripts/auto_launch_*.ps1` 有 fail-closed 串行看护，不要并发启动多项训练。

## 配置与模型锁

- `configs/pipeline*.yaml` **内容实际是 JSON**（JSON 是合法 YAML 1.2）；不要假设存在 YAML 专有语法。
- profile 解析顺序：`configs/profiles/<name>.yaml` → `src/pcbparse/resources/profiles/<name>.json`。生产 profile 仅 `baseline`/`competition`；`debug`/`dataset_*` 属调试。
- **`models/models.lock.json` 是唯一生产权威**；`models/registry.yaml` 与 `configs/models.yaml` 仅描述性，不能替代它。
- 禁止手工编辑锁的 `blessed`/`status`。默认锁为 4 任务（`component`/`ocr`/`pin`/`wire`）+ `deterministic_vectorizer` junction；仅当 `graph.learned_junctions_enabled=true` 才允许第五个 `junction` 任务。
- 训练只产出 candidate bundle，不会自动写锁；缺项/哈希不符/缺 lineage 都保持 unready（fail-closed）。

## 硬性边界

- Validator 0 ERROR 才允许 Serializer 输出；缺 component key / pin number 时**禁止自动编号、占位填充或伪造字段**绕过门禁。批量失败写 `error_manifest.json`。
- 生产推理只读图像、模型、配置、静态器件库；**禁止读同名目标 JSON/sidecar**。GT/oracle 仅限明确标记的离线诊断，不得混入正式预测或晋级证据。
- 本地评估用 `pcbparse.evaluation.raw_cli`（raw-v1）；`evaluation.cli` 是 guide-v1 回归器，**不是官方 evaluator**，`weighted_score` 不得当作竞赛成绩。官方 evaluator 尚未获得。
- 仅 `dataset300-v2` 可用于训练（240/30/30 grouped split）；`dataset300-v1` 仅供审计，禁止混用。
- 公开数据集 210 张与 Dataset300 有精确字节重复，**不能当新 holdout**；`release50` 已反复调优，只作回归，不能宣称独立泛化。
- 时间预算：单图 600s / 全批 7200s（`configs/pipeline.yaml` 的 `run` 段）。
- 输出顶层严格为 `components`/`pins`/`nets`，不得泄漏诊断/证据字段；相同输入+配置的 JSON 必须字节级确定。

## 目录职责

| 目录 | 用途 |
|---|---|
| `src/pcbparse/` | 活跃代码（`cli.py` 为 `pcbparse` 入口） |
| `data_tools/` | 数据/晋级 CLI（独立包，自举 src） |
| `tests/unit` `tests/integration` | unittest 用例 |
| `work/` | 可复现实验输出（新实验写新子目录） |
| `artifacts/` | 持久审计证据 |
| `archive/` | 只读历史，禁止新实验 |
| `external/` | 官方 Docker 等外部资产，禁止原地修改 |
| `outputs/` `output/` | 用户侧推理/报告输出 |

## 文档导航与时效

- `docs/Agent构建执行说明_2026-09-22.md` — 构建任务约束与交付格式（重要边界清单）。
- `docs/PROJECT_STRUCTURE.md` — 目录分区与合并核对。
- `docs/implementation_status.md` — 历史状态矩阵与复现命令。
- README 与 implementation_status 中的指标是**历史快照**；与 `models/models.lock.json` 或当前配置冲突时，**以锁和可执行配置为准**。
