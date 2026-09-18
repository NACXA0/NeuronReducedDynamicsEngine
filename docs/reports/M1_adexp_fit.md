# M1 — AdExp 拟合对比

**状态**：骨架（数值由 `scripts/make_figures.py` 与 `nrde fit` / `pytest` 填充）  
**对应**：RFC-001 验收条 1；NFR-4；`tests/test_fit.py`

## 协议

- 模型：AdExp（主对象）；ExpLIF / LIF 为对照
- 观测量：稳态 f-I 曲线 \(r(I)\)
- 拟合：PCHIP（CubicSpline 过冲则降级）；质量门 `r2 ≥ 0.98`
- 产物：`artifacts/aCC.npz` + `aCC.meta.json`（含 `type_id`、`layer`、`git_commit`、`config_hash`）

## 结果摘要

| 指标 | 目标 | 实测 |
|---|---|---|
| \(R^2\) | ≥ 0.98 | （运行 `nrde fit --model adexp` 后填入） |
| MSE | 报告 | |
| 方法 | pchip / cubic | |

```bash
nrde offline fit --model adexp --type-id aCC
python scripts/make_figures.py --models adexp
```

## 图

- `figures/M1_adexp_fi.csv` — `I_nA,r_ode_Hz,r_F_Hz`
