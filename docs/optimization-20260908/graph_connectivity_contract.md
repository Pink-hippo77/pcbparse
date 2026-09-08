# 图连接契约 — 2026-09-08

## 连接类型定义

构建器和校验器共享以下连接判定。校验器独立重算几何，不信任构建器的 union ID。

### 1. 端点接触 (Endpoint Touch)

两条同通道线段的端点距离 ≤ `endpoint_merge_tolerance_px`（默认 3.0px）。

### 2. 同通道共线重叠 (Collinear Overlap)

两条同通道线段投影到支撑线后有重叠，且偏移 ≤ `collinear_merge_tolerance_px`。

- 需要 `collinear_merge_tolerance_px > 0` 才启用
- 校验器用相同容差独立判定
- 默认配置：0.0（禁用）；release-candidate：0.75

### 3. 共线间隙合并 (Collinear Gap)

两条同通道共线线段之间有间隙，但 `gap + max_offset ≤ collinear_gap_merge_tolerance_px`。

- 需要 `collinear_gap_merge_tolerance_px > 0` 才启用
- 要求真实像素证据（不能无条件跨空白连线）
- 默认配置：0.0（禁用）；release-candidate：6.0

### 4. 语义 Junction 连接

`kind` 为 `t_junction`、`connected_cross` 或 `bend` 的 Junction 连接其 `segment_ids`。

### 5. Source Edge/Polyline 连接

同一 `source_edge_id` 的线段通过端点接触判定连接。同一 `source_polyline_id` 的线段也通过端点接触判定——不信任远距离复用 ID。

### 6. Golden/Dataset 连接

`source="dataset"` 的证据可跳过几何检查，直接信任。

## 校验器行为

- 默认调用（无容差参数）：只检查端点邻接 + Junction + source ID → 旧行为
- 带容差调用：额外检查共线重叠/间隙 → 与构建器一致
- Pipeline orchestrator 从 config 读取容差并传递给校验器

## 测试覆盖

| 测试 | 文件 |
|---|---|
| `test_collinear_overlap_no_false_positive_with_matching_tolerances` | `test_p0_contract_fixes.py` |
| `test_collinear_overlap_still_fails_without_tolerance` | `test_p0_contract_fixes.py` |
| `test_collinear_gap_no_false_positive_with_matching_tolerances` | `test_p0_contract_fixes.py` |
| `test_distant_segments_same_channel_two_nets` | `test_p0_contract_fixes.py` |
| `test_distant_reused_polyline_not_merged_by_builder` | `test_p0_contract_fixes.py` |
| `test_non_connected_cross_remains_two_physical_nets` | `test_graph_fixtures.py` |
| `test_distant_reused_polyline_id_does_not_create_giant_net` | `test_graph_fixtures.py` |
| `test_endpoint_classification_never_bridges_channel_separated_segments` | `test_graph_fixtures.py` |
