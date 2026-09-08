# Runtime/CUDA 审计 — 2026-09-08

## 硬件环境

| 项目 | 值 |
|---|---|
| CPU | i5-12500H, 12 核 / 16 逻辑处理器 |
| RAM | 16 GB |
| GPU | RTX 3050 Ti Laptop, 4 GB VRAM |
| CUDA | 12.6 |
| cuDNN | 9.1.0.70 |

## PyTorch 环境

| 项目 | 值 |
|---|---|
| PyTorch | 2.12.1+cu126 |
| CUDA available | True |
| GPU 可见 | NVIDIA GeForce RTX 3050 Ti Laptop GPU |

**结论**: PyTorch CUDA 正常工作。

## ONNX Runtime 环境

| 项目 | 值 |
|---|---|
| ORT 版本 | 1.28.0 |
| 可用 providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| CUDA provider 实际可用 | **否** — 缺少 `cublasLt64_13.dll` (CUDA 13) |

**问题**: ORT 1.28.0 默认依赖 CUDA 13，而本机是 CUDA 12.6。`get_available_providers()` 报告 CUDA 可用，但创建 CUDA session 时静默回退到 CPU。

**错误日志**:
```
FAIL: Error loading "onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing.
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*
```

## 兼容性矩阵

| ORT 版本 | CUDA 要求 | cuDNN 要求 | 本机兼容 |
|---|---|---|---|
| 1.20.x | CUDA 12.x | cuDNN 8.x/9.x | ✅ 可能 |
| 1.21.x | CUDA 12.x | cuDNN 8.x/9.x | ✅ 可能 |
| 1.22+ | CUDA 12.8+ / CUDA 13 | cuDNN 9.x | ❌ |
| 1.27+ | CUDA 13 | cuDNN 9.x | ❌ |
| 1.28.0 (当前) | CUDA 13 | cuDNN 9.x | ❌ |

## 已实施的修复

### 1. ONNX session provider 验证 (P1-A)

**文件**: `src/pcbparse/perception/torch_pin_adapter.py`

在 `_ONNXBackend.__init__` 中，创建 session 后验证 `session.get_providers()` 是否包含请求的 provider。显式 CUDA 请求但实际回退时抛出明确错误。

### 2. OCR batch size 可配置

**文件**: `src/pcbparse/perception/production_provider.py`, `composite_provider.py`

OCR 的 `global_batch_size`、`component_batch_size`、`strip_batch_size` 现在从 pipeline config 的 `ocr` section 读取，不再硬编码。

配置示例:
```json
{
  "ocr": {
    "global_batch_size": 8,
    "component_batch_size": 4,
    "strip_batch_size": 16
  }
}
```

### 3. Perception 子阶段计时

**文件**: `src/pcbparse/perception/composite_provider.py`

`build_ir` 现在记录各阶段耗时到 `ir.metadata["phase_timings_seconds"]`:
- `image_load`: 图像加载
- `component_detect`: 元件检测（含 NMS、fallback）
- `ocr_recognize`: OCR 检测+识别
- `pin_detect`: Pin 检测
- `wire_segment`: 导线分割+向量化
- `total_perception`: 总计

## 未完成的工作

| 项目 | 原因 | 所需操作 |
|---|---|---|
| ORT CUDA 12 兼容版本安装 | 需要隔离环境验证 ORT 1.21.x | 创建 `.venv-ort12` 并测试 |
| Pin microbatch 配置化 | Pin 模型文件不在 git 中 | 需要模型文件才能测试 |
| GPU 训练/推理 benchmark | 需要模型文件和 CUDA ORT | 安装兼容 ORT 后执行 |
| Docker CUDA/ORT 兼容性 | 需要 Docker 环境 | 目标 Linux 机器验证 |

## 建议下一步

1. **安装 ORT 1.21.x 到隔离环境**:
   ```bash
   python -m venv .venv-ort12
   .venv-ort12\Scripts\pip install onnxruntime-gpu==1.21.1
   ```
   验证 CUDAExecutionProvider 实际可用。

2. **将 Pin microbatch 参数化**: 在 `PinAdapterConfig` 中添加 `microbatch_size` 字段，从 model lock 的 inference section 读取。

3. **在 GPU 可用时运行 release50 smoke test**: 验证完整流水线在 GPU 上的耗时和显存。
