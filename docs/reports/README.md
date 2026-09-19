# M1 验证报告索引

M1 交付物三件套落点（RFC-001 §4）：

| 报告 | 文件 | 内容 |
|---|---|---|
| 拟合对比 | [M1_adexp_fit.md](M1_adexp_fit.md) | AdExp f-I vs 在线 ODE，NFR-4 |
| 在线 vs 降阶 | [M1_online_vs_reduced.md](M1_online_vs_reduced.md) | VP / 速率误差 |
| chirp \|Z(ω)\| | [M1_chirp_impedance.md](M1_chirp_impedance.md) | 频域峰值与层级判定 |

生成数值曲线（CSV）：

```bash
uv run python scripts/make_figures.py --out docs/reports/figures
```
