# M1 — AdExp 拟合对比

**状态**：已用默认数据集重算（[证据链](evidence.md)）  
**对应**：RFC-001 验收条 1；NFR-4

## 协议

- Ground truth：在线 AdExp Euler，`dt = 0.05` ms
- 表：`datasets/fly/fits/aCC_l0.npz`（稳态 f-I；meta `layer=L1`，因适应）
- 对照：`datasets/fly/fits/default_inhibitory_l0.npz`（统一 LIF）
- 电流：训练网格中点（留出），`t_total=800` ms，窗口 `400` ms

## 结果摘要

| 指标 | 目标 | 实测 |
|---|---|---|
| \(F\) vs ODE \(R^2\) | ≥ 0.98 | **0.995** |
| \(F\) RMSE | 报告 | 1.6 Hz |
| LIF RMSE（同一 ODE） | 对照 | 110 Hz |
| 方法 | pchip | pchip（meta） |

图：`figures/adexp_fi_overlay.svg`（CSV 同名）。不要在本页手填新数；改协议后跑：

```bash
uv run --no-sync python scripts/make_figures.py
```
