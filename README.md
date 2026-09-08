# PCB Schematic Parser

这是依据《赛题六：AI 驱动的 PCB 原理图自动解析》实施手册落地的可执行基线。系统以 `Circuit-IR` 为唯一内部表示，严格分离感知、几何关联、Physical Graph、Electrical Net、验证和官方 JSON 序列化。

当前版本完成了工程闭环、官方 Dataset300 数据底座、真实标注转换、五类训练清单、图拓扑、Validator、确定性 Serializer、CLI 与 HTTP 接口。它包含四种明确区分的运行方式：

- `production`：`baseline`/`competition` 的默认入口；仅从通过逐文件哈希、模型包 lineage 和元数据门禁的模型锁加载精确 task 集合。当前默认是 `component`/`ocr`/`pin`/`wire` 四类 adapter，junction 使用确定性 vectorizer；显式开启 learned junction 才增加第五类。任何缺项都会 fail-closed。
- `dataset_golden`：读取 Dataset300 配套标注，做严格集成/拓扑回归；缺失字段绝不猜测，因此部分样本会被正式 Serializer 阻断。
- `dataset_debug`：仅用于接口与端到端调试；用带 `DATASET_` 前缀的透明占位符补足缺失/冲突的官方 key，不得作为模型精度或正式提交结果。
- `heuristic`：只看图片的低依赖 fast path，仅用于线路烟测；它不会使用当前配置的学习模型，也不能代表候选精度。

## 当前验收结论

截至 2026-08-27，主模型锁已经切换为可移植的 blessed candidate-bundle 发布锁；
`models/models.lock.json` 中的文件、适配器、推理配置和 bundle lineage 均通过校验。
`production_ready` 仍保持 false，因为正式 50-case、3 次重复、性能余量和最终精度门禁尚未完成。普通 `healthcheck` 只证明进程存活；
`healthcheck --require-ready --config configs/pipeline.yaml --models-lock models/models.lock.json` 当前返回 exit code 0；runtime ready 不等于竞赛发布 ready，生产入口仍会对不合格图像 fail-closed。
冻结验证集的关键结果为：

| 任务 | 当前证据 | 判断 |
|---|---|---|
| Component | mAP50 0.2124，P/R 0.3667/0.2338 | 未过 0.25 门禁 |
| OCR | aggregate exact 0.6785、CER 0.1744；designator exact 0.9038（n=364）、pinnumber exact 0.7569（n=650）；detector P/R/Hmean 0.5968/0.3250/0.4208 | 字段级 recognizer 证据已补齐，detector 与严格 E2E 仍失败；0024 冻结视觉键候选为 0/15 exact、0 赋键 |
| Pin | 旧权重按现行 decoder 重算 P/R/F1@4px 0.4249/0.5189/0.4673，pred/GT 1.221；v4 在完整 epoch 7 后外部中止 | 无正式终态 bundle；boundary-band 验证虽得 F1 0.5786，但 recall 0.4496、pred/GT 0.554，仅可用于 v5 设计 |
| Wire | skeleton F1 0.8560，connectivity 0.5811 | 任务级候选通过，最终 Graph 未达标 |
| Junction | 六类 macro-F1 0.4280；有支持三类 macro-F1 0.8560 | 真实六类门禁失败，三类 val support 为 0 |

单例严格 image-only 联调仍会因缺 component key、缺 pin number 和断裂 net
被 Validator 阻断，因此不能声称已达到高度可用竞赛水平。完整矩阵、指标口径、
训练状态与复现命令见
[实施状态](docs/implementation_status.md)和
[系统完成度分析](docs/系统完成度分析与下一步方案.md)。机器可读的时间点快照见
[完成度与就绪审计](artifacts/audits/completion_readiness_audit.json)。

## 环境

- Python 3.11 或 3.12
- 运行依赖：NumPy、Pillow、Pydantic
- 官方数据根：`Dataset300/Dataset300`
- 训练模型为可插拔 Provider；工作区包含正式候选包和本地预训练初始化文件，
  运行时不会隐式联网下载。当前主锁绑定 `models/releases/v8-portable` 下的可移植 runtime bundle；正式 release gate 通过前，不能把 runtime-ready 误称为 production-ready。

在本项目已配置的 Codex 环境中，可使用捆绑 Python：

```powershell
$env:PYTHONPATH=(Resolve-Path '.\src').Path
$python='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python -m pcbparse.cli healthcheck
```

常规环境安装：

```powershell
py -3.11 -m venv .venv
$python=(Resolve-Path '.\.venv\Scripts\python.exe').Path
& $python -m pip install -e .
& $python -m pcbparse.cli healthcheck
```

Linux/macOS：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pcbparse.cli healthcheck
```

## 数据底座

生成清单并审计：

```powershell
& $python -m data_tools.pcbnet_ingest `
  --dataset-root .\Dataset300\Dataset300 `
  --output .\data\manifests\dataset300.jsonl `
  --expected-count 300

& $python -m data_tools.audit `
  --dataset-root .\Dataset300\Dataset300 `
  --output .\artifacts\audits\dataset300_audit.json
```

审计会把已知的 8 个图像扩展名/真实格式不一致和 11 个重复/近重复/同工程族组报告为 warning。`--strict` 将任何 warning 视为非零退出，适合发布门禁；当前原始官方数据使用 strict 会预期失败，但普通审计为通过。

生成零泄漏的 grouped split：

```powershell
& $python -m data_tools.split_grouped `
  --manifest .\data\manifests\dataset300.jsonl `
  --output .\data\splits\dataset300_split.jsonl `
  --seed 20260812
```

生成五类训练清单：

```powershell
& $python -m data_tools.derive_datasets `
  --src .\Dataset300\Dataset300 `
  --out .\data\derived\dataset300-v2 `
  --version dataset300-v2 `
  --split-manifest .\data\splits\dataset300_split.jsonl
```

`dataset300-v2` 是当前唯一可用于训练的派生版本：它绑定
`dataset300_split.jsonl`（240/30/30），并把精确重复与感知近重复样本固定在同一
split。仓库中早期的 `dataset300-v1` 和 `dataset300_split.json` 只为审计追溯保留，
不得与 v2 混用。

产物包括 `component_detection.jsonl`、`pin_keypoints.jsonl`、`ocr_text.jsonl`、`wire_paths.jsonl`、`junction_patches.jsonl` 和 `summary.json`。Wire 清单只保存惰性几何引用，不复制 437 万个 dense coordinate 点。

训练前先验证整个数据 lineage（文件哈希、300 个样本、240/30/30 split、重复组零泄漏和 147,758 条任务记录）：

```powershell
& $python -m pcbparse.training.preflight_cli `
  --config .\configs\train\dataset300_v2_torch.json `
  --workspace-root .
```

`dataset300_v2_torch.json` 保留为五任务通用训练基线；当前优化使用
`dataset300_v2_pin_v5_boundary.json`、`dataset300_v2_component_v4.json` 与
`dataset300_v2_ocr_v3_shrink_warmstart.json`，并按 Pin → Component → OCR 串行执行。
Component 初始化权重以本地路径和 SHA-256 固定，训练过程不会隐式联网。
先用不可登记的 bounded smoke 验证真实 GPU、验证、导出和 adapter 链路：

```powershell
$trainPython=(Resolve-Path '.\.venv-train\Scripts\python.exe').Path
& $trainPython -m pcbparse.training.torch_train_cli `
  --config .\configs\train\dataset300_v2_torch.json `
  --task wire --output .\work\wire-smoke --workspace-root . `
  --device cuda --precision fp16 --smoke
```

去掉 `--smoke` 才是正式训练；它只在完整配置策略、验证门禁和导出一致性
全部通过后生成 candidate，仍不会自动写入或 bless `models.lock.json`。
Junction 训练器已经实现，但官方可信标注对 connected cross、
non-connected cross 与 noise 的真实验证 support 均为 0；synthetic 标签仅允许
进入 train，不能替代真实六类门禁。2,283 条 UNKNOWN cross 记录必须经过人工
裁决、重物化和重训。`dataset300_v2_tiny.json` 仅是 fail-closed 契约夹具，
不得用于精度声明。

## 推理与接口调试

Dataset300 单例端到端调试：

```powershell
& $python -m pcbparse.cli infer `
  --input .\Dataset300\Dataset300\0001\0001-KiCad.png `
  --output .\outputs\debug `
  --profile debug `
  --provider dataset_debug `
  --fail-fast
```

严格 golden 回归用于暴露数据字段缺失；出现 `component key`、`pin number` 阻断是正确行为：

```powershell
& $python -m pcbparse.cli infer `
  --input .\Dataset300\Dataset300\0001\0001-KiCad.png `
  --output .\outputs\strict `
  --profile debug `
  --provider dataset_golden `
  --fail-fast
```

`baseline` 与 `competition` 是生产 profile：模型锁未通过时默认 fail-closed，不会创建输出。图像-only 启发式链路仅可通过显式调试开关运行，返回状态为 `degraded`、执行模式为 `unready_debug`，不得作为精度或提交结果：

```powershell
& $python -m pcbparse.cli infer `
  --input .\Dataset300\Dataset300\0001\0001-KiCad.png `
  --output .\outputs\heuristic `
  --profile baseline `
  --provider heuristic `
  --allow-unready-debug
```

发布门禁可独立检查：`& $python -m pcbparse.cli healthcheck --require-ready`。模型锁必须覆盖当前配置声明的精确 task 集合：本仓库默认关闭 learned junction，使用确定性 wire vectorizer，因此是 `component`、`ocr`、`pin`、`wire` 四个 task；只有显式开启 `graph.learned_junctions_enabled` 时才允许增加 `junction`。每项必须包含相对模型锁目录解析的 `path`、真实文件 `sha256`、`status: "blessed"`、`blessed: true`、`package.module:factory` 形式的 `adapter`，以及非占位的 `version`、`license` 和受支持的 `precision`（`bf16`、`float16`、`float32`、`fp16`、`int4`、`int8`、`mixed` 或 `uint8`）。空锁、空模型文件、缺失/重复/未知 task、重复 JSON key、文件缺失或哈希不符都会保持 unready。批量 infer 失败时会在输出目录生成 `error_manifest.json`，CLI 汇总也会返回该路径；strict 模式不会用自动编号或启发式占位符掩盖缺失的器件标识。

`--require-ready` 不只检查元数据：哈希和 lineage 验证通过后还会实际导入当前配置所需的 adapter、加载 provider，并对固定图像执行一次感知 smoke；导入、加载或协议调用失败都会把 readiness 降为 unready。

HTTP 服务只接受配置输入根下的相对路径，不上传任意文件：

```powershell
& $python -m pcbparse.cli serve `
  --host 127.0.0.1 --port 8080 `
  --input-root .\Dataset300\Dataset300 `
  --output-root .\outputs\api `
  --profile debug --provider dataset_debug `
  --allow-unready-debug
```

接口：

```text
GET  /health
POST /v1/infer
Content-Type: application/json
{"input":"0001/0001-KiCad.png","output":"0001.json"}
```

该 HTTP 层用于本地/受信网络调试，不应直接暴露到公网。公网部署须在前置网关配置 TLS、认证、限流与访问日志；`/health` 是存活探针，`/ready` 在五类 blessed 权重未登记时返回 503。未就绪时 `/v1/infer` 默认也返回 503；只有服务启动者显式传入 `--allow-unready-debug` 才能开放本地调试，客户端请求无权自行绕过门禁。

## 测试与 readiness

2026-08-23 23:22（Asia/Shanghai）基于最终代码/配置的回归结果：单元测试
244 项（242 通过、2 跳过），集成测试 39 项全通过；合计 283 项、0 失败。
普通健康检查 exit `0`，严格就绪检查 exit `1`。测试通过证明工程契约和
fail-closed 行为正确，不代表五个模型已达到精度门禁。

```powershell
$env:PYTHONPATH=(Resolve-Path '.\src').Path
$trainPython=(Resolve-Path '.\.venv-train\Scripts\python.exe').Path
& .\scripts\run_tests.ps1 -Python $trainPython
& $trainPython -m pcbparse.cli healthcheck
& $trainPython -m pcbparse.cli healthcheck --require-ready
```

关键发布门禁：

- 坐标变换 round-trip 最大误差不超过 0.5 px。
- Validator 为 0 个 ERROR 后才允许 Serializer 运行。
- 官方输出顶层严格为 `components`、`pins`、`nets`，且无诊断/证据泄漏。
- 相同环境、输入和配置重复输出的 JSON 字节与 SHA-256 相同。
- SIGNAL 标签可配置合并；POWER/GROUND 默认禁止通用同名合并。
- Qwen 默认关闭，只允许在候选集合中选择或 `UNKNOWN`；没有端点配置就不会联网。

本地精度评估使用 raw-v1 掩码契约，直接读取原始 CVJsonStd 并对缺失/歧义字段 mask；预测侧必须是持久化 Circuit-IR，且 manifest 显式声明 `sidecar_access:false`。评估器会从 provider、metadata 和 evidence 多层拒绝 `dataset_golden`/`dataset_debug` 泄漏：

```powershell
& $python -m pcbparse.evaluation.raw_cli `
  --prediction-manifest .\artifacts\predictions\test.jsonl `
  --dataset-root .\Dataset300\Dataset300 `
  --split-manifest .\data\splits\dataset300_split.jsonl `
  --split test `
  --output .\artifacts\evaluations\raw-v1-test.json
```

raw-v1 报告 Component/Pin/Wire/Graph 的独立指标与 eligible/masked 覆盖率，不自创竞赛总分。旧 `evaluation.cli` 仅是 `guide_v1_assumption_regression` 格式回归器，明确不是官方 evaluator，也不得用其 `weighted_score` 宣称竞赛成绩。

## 已知边界与下一阶段

当前优先级不是“把五个文件放进锁里”，而是补齐可 promotion 的质量证据：

1. 等待正在运行的 Pin v5 正式终态；其配置 SHA-256 为
   `7b966fc7…ac90b4`，使用 `boundary_band_v2` 和冻结 val 选择的阈值 `0.75`，
   但中间轮次与 smoke 均不可登记；
2. Pin v5 只有在 `candidate_evidence_ready` 或 `gate_failed` 终态后，才由串行
   看护启动 Component v4；
3. Component v4 终态后再启动 OCR v3。OCR v3 必须同时检查 detector Hmean、
   aggregate exact、designator exact 与 pinnumber exact，不能只看总 exact；
4. 完成 Junction 真实六类人工标注、重物化与重训；
5. 在固定 image-only 分层套件上做到 Validator 0 ERROR、Serializer 全通过；
6. 用恰好五个 verified bundle 和 Junction human-review evidence 运行
   `data_tools.model_promotion`，禁止手工 bless；
7. 获得官方 `sample_submission`/evaluator 后冻结格式与评分口径，并在目标
   Docker/GPU 环境完成 30–50 例总时长、P95、显存和确定性演练。

本地 raw-v1 与 guide-v1 都不是官方 evaluator；在官方脚本缺失时只能报告分项
指标和已知边界，不能给出官方分数或保证竞赛名次。

容器说明：默认启动 HTTP 服务并将 Docker 健康探针指向 `/ready`。在五类 blessed 权重尚未登记时，容器会有意保持 `unhealthy`，这不是进程崩溃，而是生产就绪门禁生效。完成模型注册后使用：

```bash
docker build -t pcbparse:local -f docker/Dockerfile .
docker run --rm -p 8080:8080 \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/output:/data/output" \
  pcbparse:local
```
