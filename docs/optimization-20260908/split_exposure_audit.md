# Split 曝光审计 — 2026-09-08

## 数据集结构

| Split | 样本数 | 角色 |
|---|---|---|
| train | 240 | 训练集 |
| test | 30 | 原始独立测试集（**已曝光**） |
| val | 30 | 验证集 |
| **总计** | **300** | |

## release50 曝光历史

| 属性 | 值 |
|---|---|
| release50 样本数 | 50 |
| 来源 test | 30（100% 的原始 test） |
| 来源 val | 20（67% 的原始 val） |
| 用途 | 多次阈值选择、参数扫描、回归测试 |
| 当前角色 | **已曝光回归集**（regression & performance exercise） |
| 独立性 | **已失效** — 不能声称独立 test |

## 证据

```python
# dataset300_split.jsonl 分布
{'train': 240, 'test': 30, 'val': 30}

# dataset300_release50_v1.jsonl 来源
# 所有 50 条标记为 split='release50'
# 交叉引用原始 split:
#   test: 30, val: 20
```

## 约束

1. **不能重新抽取旧数据伪装未曝光 holdout** — release50 中的 30 张 test 已被反复用于选参
2. **后续参数选择只在开发集进行** — 开发集应从未曝光数据中构建
3. **新独立数据到位前继续工程工作** — 不伪造独立性
4. **GT 覆盖不完整** — 650/2155 元件有 designator，1508/5163 引脚有 number；缺标不能当负样本

## 后续行动

- [ ] 新增未曝光工程族或后续独立采集建立外部 holdout
- [ ] 官方 evaluator 到位后做独立确认
- [ ] 保留此审计作为数据 lineage 记录
