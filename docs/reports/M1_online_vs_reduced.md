# M1 — 在线 ODE vs 降阶 F

**状态**：骨架  
**对应**：RFC-001 验收条 2；NFR-5；`tests/test_validation.py`

## 协议

- 单胞：同一电流协议下 ODE 尖峰列 vs 降阶 F（L0/L1/L2）
- 度量：Victor–Purpura（`q = 1/10 ms`）或相对速率误差
- 不达标类型写入 [known-inapproximable.md](../known-inapproximable.md) 并允许局部 ODE 回退

## 结果摘要

| 模型 / 类型 | 层级 | VP / 相对误差 | 判定 |
|---|---|---|---|
| AdExp | L1/L2 | （填） | |
| ExpLIF | L0 | （填） | |

## 复现

```bash
uv run --extra dev pytest tests/test_validation.py -q
```
