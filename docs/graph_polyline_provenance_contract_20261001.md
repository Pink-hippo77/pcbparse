# 重复线段的折线来源保留契约

2026-10-01。补充既有 Physical Graph 连接契约。

`WireSegment.source_polyline_id` 继续表示主要观测折线。新增内部字段
`source_polyline_ids`，用于精确同通道、同端点线段去重后保留全部已观测折线
来源。旧 IR 缺少该字段时使用空列表。`observed_polyline_ids` 返回主来源与
列表来源的排序并集，不进入官方 JSON。

去重仅在精确几何和 channel 相同的组内合并来源；不合并相邻线、不同通道
或近似重复的坐标。选择 canonical segment 的 score/id 规则保持不变，原图
坐标、pin 身份与网络成员不因来源合并而改写。移除重复线段的 RepairRecord
仍保存原始线段及来源，供追溯。

建图器和验证器都通过 `collect_segment_connections` 使用上述来源并集。
同源只提出候选，仍必须满足原有端点接触容差；远距离复用来源不得形成连接。
不同 channel 的折线转角沿用原有同源接触规则，不新增交叉或缺口接受规则。

生产执行的 `degraded` 模式可在建图前裁去不可表示的无身份对象，但不得
将验证器的 ERROR 降为 WARNING 来通过 Serializer。仅明确 debug 执行保留
原有诊断输出行为。生产严格复核有 ERROR 时，整图失败并由批处理记录。

验收覆盖去重前后转角连接、来源远距复用、通道隔离、排序确定性、修复幂等，
以及 degraded 生产模式的严格门禁。开发验证不能代替赛题官方四项评测。
