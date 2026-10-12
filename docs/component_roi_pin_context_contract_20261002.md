# 器件裁剪区域引脚文字上下文

`association.pin_body_component_roi_context_enabled` 默认 `false`，只在
`pin_body_name_context_enabled=true` 时可启用。两个配置项都要求真实布尔值；
依赖缺失或字符串、数字代替布尔值时拒绝配置。

此开关允许现有 `BodyPinLabelContext` 检查 component 级 OCR 裁剪中的未分类文字。
检查仍要求实际器件边界、已感知引脚、对应行列方向，以及同一行可见的数字标签。
数字标签已声明其他器件归属时不参与匹配。仅调整文字的角色分类，随后由原有
关联器完成归属匹配；不会填充器件 key、引脚 number/name 或创造引脚坐标。

已有明确类型、dataset 证据、传统器件位号、器件型号和 side_strip 路径受到保护。
原始文字及其大小写保留。开关关闭时使用原有分类路径。生产配置没有开启此开关，
它目前用于可复核的开发对照；候选 OCR 和器件身份规则仍未晋级生产模型锁。

2026-10-02 的同一批暴露过的 grouped val30 冻结感知对照：两组均 30/30 输出，
严格验证 ERROR 均为 0；开启开关后的 30 份官方格式 JSON 与上一轮私有原型逐字节一致。
raw-v1 离线引脚名称精确匹配由 114/611 提升到 172/611。这不是官方竞赛分数，
也不能据此认定器件、引脚、网络、线段四项达到 95%。

证据目录：`work/optimization-cycle11-server-gpu-20261002`。
运行 `native_replay30.py` 后，再运行 `evaluate_native.py`；GT 只在第二步离线读取。
完整 unittest 先 unit 后 integration，共 803 项通过、71 项跳过。
