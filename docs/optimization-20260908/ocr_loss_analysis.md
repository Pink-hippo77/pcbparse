# OCR 损失分解分析 — 2026-09-08

## 工具

`scripts/ocr_loss_decomposition.py` — 从缓存 IR 中逐 token 追踪 OCR 损失。

### 损失阶段定义

| 阶段 | 含义 | 典型原因 |
|---|---|---|
| none | 通过所有阶段 | 正常 |
| recognizer | 文字框检出但识别失败 | 识别器输出为空或异常 |
| semantic | 识别成功但无法分类 | 规则分类器无法确定类型 |
| owner | 已分类但无有效 owner | 空间距离超限或无匹配元件 |
| matching | 有 owner 但关联规则拒绝 | 正则不匹配、margin 不足 |
| conflict | 多候选竞争失败 | 同一 key 被更高分候选占据 |

### 使用方法

```bash
# 需要先有缓存 IR（从 strict run 或 production eval 产出）
python scripts/ocr_loss_decomposition.py \
  --ir-dir work/production-e2e-componentmargin0-pinmargin16/ir \
  --split data/splits/dataset300_release50_v1.jsonl \
  --gt-dir Dataset300/Dataset300 \
  --output artifacts/audits/ocr_loss_decomposition.json
```

**注意**: 当前 work 目录中的 IR 文件 `ir_path` 为 null（严格失败时未保存）。Prompt 0 的修复已实现 `partial_ir` 保存机制，下次运行将自动保存失败 IR。

## 已知 OCR 指标（来自历史产物）

| 指标 | 值 | 说明 |
|---|---|---|
| detector Hmean | 0.3669 | 文字框检测 |
| recognizer exact | 0.5307 | 识别精确匹配 |
| designator exact | 114/650 = 17.54% | 元件编号 E2E |
| pin number exact | 72/1508 = 4.77% | 引脚编号 E2E |

## 预期损失分布（基于代码分析）

根据 `semantic_fields.py` 的规则链和 `text_component.py` 的关联逻辑：

1. **Detector loss** (~35-40%): 大量小文字框未检出（Hmean=0.37）
2. **Semantic classification loss** (~10-15%): 非标准文本无法分类
3. **Owner assignment loss** (~15-20%): 空间距离超限或元件漏检
4. **Matching/conflict loss** (~10-15%): margin 不足或 key 冲突

## 关键瓶颈

根据审查文档中的严格运行数据：
- 1,699 个 `COMPONENT.MISSING_KEY` — 大量元件没有获得 key
- 1,681 个 `NET.DISCONNECTED_SEGMENTS` — 图连接问题（P0-A 已修复）
- designator exact 仅 17.54% — 主要损失在 detector 和 owner 阶段

## 下一步

1. 等待有缓存 IR 后运行 `ocr_loss_decomposition.py` 获得实际损失分布
2. 根据实际损失决定优先改进 detector 还是 owner
3. 如果 detector 是主损失 → 需要训练改进（Prompt 5）
4. 如果 owner 是主损失 → 可以通过几何增强和距离阈值调整改进
