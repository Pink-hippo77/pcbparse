# 优化交付总结 — 2026-09-08 至 2026-09-09

## 执行摘要

基于 `docs/架构阻塞与硬件优化审查_2026-09-08.md` 的分析，完成了 7 个 commit 的系统性优化，覆盖 Prompt 0-4 + 集成准备。435 个单元测试全部通过，70 个跳过（需要模型文件或 GPU）。

## Git 历史

```
f31be4d integration: .dockerignore, rc config OCR batch params, baseline update
673c668 prompt4: OCR loss decomposition framework and analysis
7ee5c57 prompt3: runtime audit, OCR batch config, perception sub-stage timing
71aa157 prompt1+2: targeted tests, duplicate designator fix, connectivity contract
0286c96 prompt0: baseline, status, split audit, failure IR preservation
65b9901 baseline: P0 contract fixes + prompt0 init (2026-09-08)
```

## 修改文件清单

### 核心修复

| 文件 | 修改 | 原因 |
|---|---|---|
| `src/pcbparse/graph/physical.py` | `segments_have_collinear_overlap` 提升为公共函数 | P0-A: 让校验器复用构建器的共线判定 |
| `src/pcbparse/validation/graph.py` | `validate_graph` 新增共线容差参数；`_has_disconnected_segments` 增加共线检查 | P0-A: 校验器与构建器连接判据一致 |
| `src/pcbparse/association/text_component.py` | 新增 `_is_valid_designator`、`_ALL_LETTER_DESIGNATOR`；重复标签支持 | P0-B: 接受 RUN/VCC 等全字母标签；允许重复 R1 |
| `src/pcbparse/perception/torch_pin_adapter.py` | ONNX session provider 验证 | P1-A: 显式 CUDA 请求但实际回退时明确报错 |

### 基础设施

| 文件 | 修改 | 原因 |
|---|---|---|
| `src/pcbparse/pipeline/orchestrator.py` | Pipeline 传递图容差给校验器；失败 IR 附加到异常 | Prompt 0+1: 修复验证阶段、失败可回放 |
| `src/pcbparse/evaluation/infer_manifest.py` | 严格失败时保存 partial_ir | Prompt 0: 失败案例可调试 |
| `src/pcbparse/perception/composite_provider.py` | OCR batch 配置化；perception 子阶段计时 | Prompt 3: GPU 资源控制、性能诊断 |
| `src/pcbparse/perception/production_provider.py` | 从 pipeline config 读取 OCR batch sizes | Prompt 3: 配置化 |
| `configs/pipeline.release-candidate.yaml` | 添加 OCR batch size 参数 | Prompt 3: release-candidate 配置完整 |

### 文档与审计

| 文件 | 内容 |
|---|---|
| `docs/optimization-20260908/baseline.md` | 项目基线、模型状态、配置差异、复现命令 |
| `docs/optimization-20260908/authoritative_status.json` | 四模型状态、split 曝光、严格运行历史 |
| `docs/optimization-20260908/split_exposure_audit.md` | release50 曝光记录 |
| `docs/optimization-20260908/graph_connectivity_contract.md` | 6 种连接类型形式化定义 |
| `docs/optimization-20260908/runtime_audit.md` | CUDA/ORT 兼容性审计 |
| `docs/optimization-20260908/ocr_loss_analysis.md` | OCR 损失分解分析 |
| `docs/optimization-20260908/p0_contract_fixes_audit.md` | P0 修复前后对比证据 |
| `docs/optimization-20260908/delivery_summary.md` | 本文件 |

### 测试

| 文件 | 测试数 |
|---|---|
| `tests/unit/test_p0_contract_fixes.py` | 20 个（图连接 + 命名契约） |
| `tests/unit/test_ocr_loss_decomposition.py` | 8 个（OCR 损失追踪） |
| `scripts/ocr_loss_decomposition.py` | OCR 损失分解工具 |

### 配置与构建

| 文件 | 修改 |
|---|---|
| `.gitignore` | 排除 work/releases/Dataset300/环境/凭据 |
| `.dockerignore` | 排除 .venv-*/docs/scripts/tests/releases |

## 关键验证结果

### 合同探针（修复前后对比）

| 探针 | 修复前 | 修复后 |
|---|---|---|
| 共线重叠 → 校验器 | `DISCONNECTED_SEGMENTS` ❌ | `[]` ✅ |
| 共线间隙 → 校验器 | `DISCONNECTED_SEGMENTS` ❌ | `[]` ✅ |
| RUN 设计符 | `[null]` + FORMAT_REJECTED | `["RUN"]` ✅ |
| 重复 R1 | `["R1", null]` + KEY_CONFLICT | `["R1", "R1"]` ✅ |
| 远距离复用 polyline | 2 nets（正确） | 2 nets（正确） |

### 测试基线

```
单元测试: 435 passed, 70 skipped (100s)
探针: 全部通过
```

### 模型锁

SHA-256: `af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413` — 未修改。

## 剩余阻塞项

| 项目 | 阻塞原因 | 所需操作 |
|---|---|---|
| Prompt 5 (Component/Pin) | 需要模型文件和 GPU | 安装模型到 `releases/`，验证 CUDA ORT |
| Prompt 6 (发布证据) | 依赖 Prompt 5 | 等 Prompt 5 完成 |
| ORT CUDA 12 | ORT 1.28 需 CUDA 13，本机 CUDA 12.6 | `pip install onnxruntime-gpu==1.21.1` 到 `.venv-ort12` |
| OCR 损失分解运行 | 缺少缓存 IR（ir_path=null） | 下次 production run 自动保存 |
| CLI 集成测试 | debug provider 的 dataset_debug 路径预存失败 | 非阻塞，与生产路径无关 |

## 回滚方法

```bash
# 完整回滚到审查前
git checkout 65b9901

# 只回滚代码改动，保留文档
git checkout 673c668 -- src/ tests/ scripts/ configs/
```

## 交付格式合规

每次交付包含：修改文件、原因、实际命令、结果与分母、退化（无）、峰值资源（CPU 测试）、模型哈希（未变）、剩余阻塞、回滚方法。
