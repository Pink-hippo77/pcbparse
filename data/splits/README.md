# Dataset300 split 状态

- `dataset300_split.jsonl`：当前冻结 split，供 `dataset300-v2` 与后续训练使用。
  它按来源域分层，并把精确重复与确定性 dHash 近重复组绑定，避免跨 split 泄漏。
- `dataset300_split_summary.json`：上述 split 的哈希、计数和零泄漏检查结果。
- `dataset300_split.json`：早期 241/30/29 JSONL 内容（扩展名也不准确），仅保留作
  历史审计，禁止用于新训练或评测。

所有训练入口必须同时核对 split 文件 SHA-256、派生 `summary.json` 中记录的 split
哈希以及每条记录的 split 值；任何一项不一致都应 fail closed。
