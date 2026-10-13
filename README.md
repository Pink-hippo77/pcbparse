# PCB 原理图自动解析

面向 PCB 原理图图像的解析项目，按分层流水线识别器件、文字、引脚和导线，构建电路连接关系，并输出符合赛题格式的 JSON。系统以 Circuit-IR 作为内部表示，输出前经过严格校验。

## 处理流程

`perception` 读取图像并运行模型适配器 → `association` 绑定 OCR 文本与器件、引脚 → `graph` 构建 Physical Graph → `nets` 生成 Electrical Net → `validation` 检查结果 → `output` 写出官方 JSON。

输出 JSON 顶层包含 `components`、`pins` 和 `nets`。缺少器件标识或引脚编号时，系统会按校验策略阻止序列化，不会自动编造字段。生产推理只使用输入图像、模型、配置和静态器件库，不读取标注或同名 sidecar 文件。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/pcbparse/` | 推理流水线、Circuit-IR、验证器、序列化器、CLI 和服务接口 |
| `data_tools/` | 数据集清单、审计、派生数据及模型晋级工具 |
| `configs/` | 流水线、运行 profile 和训练配置 |
| `models/models.lock.json` | 生产模型锁，是生产模型配置的权威来源 |
| `models/releases/v8-portable/` | 锁定的生产模型包及 lineage 文件 |
| `tests/` | unittest 单元与集成测试 |
| `scripts/` | 测试和训练辅助脚本 |
| `docker/` | 官方运行镜像及提交入口 |

模型权重通过 Git LFS 存储。数据集、虚拟环境和运行输出不随仓库提供；准备数据时按项目的数据配置和训练契约操作。

## 获取与安装

克隆前安装 Git LFS。克隆后拉取模型文件：

```powershell
git lfs install
git clone https://github.com/yueryu54111-commits/pcbparse.git
cd pcbparse
git lfs pull
```

项目运行时使用 Python 3.11 或 3.12，依赖为 NumPy、Pillow 和 Pydantic。Windows PowerShell 安装示例：

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -e .
$env:PYTHONPATH = (Resolve-Path './src').Path
```

Linux/macOS 可使用相同的 editable 安装方式，并将 `PYTHONPATH` 设为仓库的 `src` 目录。训练、测试和 YOLO 工具使用不同的虚拟环境；训练环境需安装与设备匹配的 PyTorch，官方 Docker 镜像提供其 CUDA 运行时。

## 检查模型就绪状态

普通健康检查：

```powershell
python -m pcbparse.cli healthcheck
```

生产就绪检查会校验配置、模型锁、权重哈希和 lineage，并加载生产适配器执行 smoke：

```powershell
python -m pcbparse.cli healthcheck `
  --require-ready `
  --config configs/pipeline.yaml `
  --models-lock models/models.lock.json
```

当前生产锁包含 `component`、`ocr`、`pin`、`wire` 四个任务，junction 使用确定性 vectorizer。只有配置启用 learned junction 时才会需要第五个模型任务。健康检查通过表示运行就绪检查通过，不代表精度达到竞赛指标或官方评分通过。

## 运行推理

将单张图像或图像目录传给 CLI。目录会递归搜索可解码的图像文件：

```powershell
python -m pcbparse.cli infer `
  --input input/schematic.png `
  --output outputs/result `
  --config configs/pipeline.yaml `
  --profile baseline `
  --provider production `
  --models-lock models/models.lock.json `
  --fail-fast
```

推理结果写入指定输出目录。批处理遇到失败时会生成 `error_manifest.json`。生产 profile 在模型锁或运行就绪门禁未通过时会 fail-closed；`dataset_debug`、`dataset_golden` 和 `heuristic` 仅用于本地调试，不能作为生产预测或精度证据。

## 数据、训练与测试

训练数据不包含在仓库中。当前训练流程只接受符合 lineage 要求的 `dataset300-v2`；`dataset300-v1` 仅供审计。训练前先运行 preflight：

```powershell
python -m pcbparse.training.preflight_cli `
  --config configs/train/dataset300_v2_torch.json `
  --workspace-root .
```

训练先用 `--smoke` 验证设备、验证和导出流程。训练产物是 candidate bundle，不会自动改写生产模型锁；模型锁晋级由 `data_tools.model_promotion` 完成，并要求完整 bundle 与证据。

正式测试口径为 unittest，按单元测试再集成测试运行：

```powershell
./scripts/run_tests.ps1 -Python ./.venv-test/Scripts/python.exe
```

也可使用 CI 等价命令：

```powershell
$env:PYTHONPATH = (Resolve-Path './src').Path
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Docker

提交镜像基于官方的 `edarace-agent-base:ubuntu24`。本机需先准备该基础镜像，再从仓库根目录构建：

```powershell
docker build -t pcbparse:edarace -f docker/Dockerfile .
```

镜像的批处理入口为 `docker/agent_system`，不启动 HTTP 服务。推理输出和模型文件的使用应遵循相应模型卡及数据授权条款。
