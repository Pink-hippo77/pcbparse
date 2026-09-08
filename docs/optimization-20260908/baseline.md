# 开发基线 — 2026-09-08

## 基线建立者

Prompt 0 集成 agent，工作区 `E:\EDA挑战题目六`。

## Git 版本化

```bash
git log --oneline -1
# 65b9901 baseline: P0 contract fixes + prompt0 init (2026-09-08)
```

- `.gitignore` 排除 `work/`、`releases/`、`Dataset300/`、`.venv-*`
- 模型权重、训练产物、大型缓存不进入版本控制
- 源码、配置、文档、测试、审计产物全部版本化

## 模型锁状态

**SHA-256**: `af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413`

| 任务 | 模型 | 状态 | 门禁 |
|---|---|---|---|
| Component | v5-fullmix | ✅ blessed, loaded | mAP50=0.2501 ≥ 0.25 |
| OCR | ocr-resume (旧版) | ✅ blessed, loaded | exact=0.5307 ≥ 0.50 |
| Pin | v5-boundary (ONNX) | ✅ blessed, loaded | F1=0.638 ≥ 0.55 |
| Wire | v5-fresh-allgate | ✅ blessed, loaded | skeleton_F1=0.940 ≥ 0.55 |

**注意**:
- OCR 实际加载的是旧 resume 模型，不是 OCR v8（v8 的 Hmean=0.446 未过 0.45 门禁）
- Component v6 的 mAP50=0.238 也未过 0.25 门禁，未加载
- Pin ONNX 在本机因 ORT 1.28 缺 CUDA 13 DLL 而回退 CPU

## 严格运行历史

| 运行 | 成功/失败 | 耗时 | 最大错误 |
|---|---|---|---|
| production-e2e-componentmargin0-pinmargin16 | 0/50 | 891s | MISSING_KEY×1699, DISCONNECTED×1681 |
| production-e2e-pin-point-projection2 | 0/50 | 1792s | MISSING_KEY×1699, DISCONNECTED×1680 |

## P0 修复状态

| ID | 问题 | 状态 | 验证 |
|---|---|---|---|
| P0-A | 图构建/校验器连接判据不一致 | ✅ 已修复 | 共线重叠/间隙探针通过 |
| P0-B | 关联层拒绝合法命名、折叠重复标签 | ✅ 已修复 | RUN 被接受，重复 R1 都获得 key |
| P1-A | ONNX Pin 后端静默回退 CPU | ✅ 已修复 | 显式 CUDA 请求验证 session provider |

详细修复记录见 `p0_contract_fixes_audit.md`。

## 配置差异

| 参数 | 默认 pipeline.yaml | release-candidate.yaml |
|---|---|---|
| collinear_merge_tolerance_px | 缺失 (→0.0) | 0.75 |
| collinear_gap_merge_tolerance_px | 缺失 (→0.0) | 6.0 |
| endpoint_merge_tolerance_px | 3.0 | 1.0 |
| component_candidate_min_score | 0.45 | 0.3 |
| component_unique_margin | 0.1 | 0.0 |
| pin_wire_snap_max_px | 6.0 | 10.0 |

**关键发现**: 默认配置缺少共线容差参数。P0-A 修复从 config 读取这些值；默认 0.0 意味着默认 pipeline 不启用共线合并，release-candidate 启用。

## 测试基线

```
407 passed, 70 skipped, 76 subtests (97s)
```

使用 `.venv-test\Scripts\python.exe` + `PYTHONPATH=src` 运行。

## 评测集状态

- release50 = 30 test + 20 val，**已曝光**
- GT 覆盖：650/2155 元件有 designator，1508/5163 引脚有 number
- raw-v1、guide-v1、strict 三套评测分开报告
- 详细审计见 `split_exposure_audit.md`

## 复现命令

```bash
# 运行单元测试
$env:PYTHONPATH = "E:\EDA挑战题目六\src"
& "E:\EDA挑战题目六\.venv-test\Scripts\python.exe" -m pytest tests/unit/ -x --tb=short -q

# 运行合同探针
& "E:\EDA挑战题目六\.venv-test\Scripts\python.exe" "E:\EDA挑战题目六\artifacts\audits\architecture_review_20260908\reproduce_contract_findings.py"

# 运行基线验证
& "E:\EDA挑战题目六\.venv-test\Scripts\python.exe" "E:\EDA挑战题目六\artifacts\audits\prompt0_verify.py"
```

## 剩余工作

| 优先级 | 项目 | 依赖 |
|---|---|---|
| P0-C | 端到端精度差距 (designator 17.5%, pin number 4.77%) | Prompt 4 损失分解 + Prompt 5 训练 |
| P0-D | 评测集污染与评分口径 | 新独立数据 |
| P1-A | ORT CUDA 12 版本兼容性 | 手动安装 ORT 1.21.x 到 .venv-ort12 |
| P1-B | 域差异 (KiCad 薄弱) | Prompt 5 训练 |
| P1-C | 失败 IR 回放、磁盘预算 | 下次 production run 验证 partial_ir 保存 |

## Git 提交历史

```
673c668 prompt4: OCR loss decomposition framework and analysis
7ee5c57 prompt3: runtime audit, OCR batch config, perception sub-stage timing
71aa157 prompt1+2: targeted tests, duplicate designator fix, connectivity contract
0286c96 prompt0: baseline, status, split audit, failure IR preservation
65b9901 baseline: P0 contract fixes + prompt0 init (2026-09-08)
```

## 回滚方法

```bash
# 回滚到此基线
git checkout 65b9901

# 只回滚代码改动，保留文档
git checkout 673c668 -- src/ tests/ scripts/ configs/ .gitignore .dockerignore
```
