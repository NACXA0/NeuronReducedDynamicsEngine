# 验证报告索引

README 宣传数字的审计入口是 **[证据链](evidence.md)**（`figures/evidence.json` + 可复现脚本）。

M1 三件套（RFC-001 §4）现在都指向同一组默认数据集计算：

| 报告 | 文件 | 内容 |
|---|---|---|
| 证据链 | [evidence.md](evidence.md) | 协议、哈希、核验 / 重算命令 |
| 拟合对比 | [M1_adexp_fit.md](M1_adexp_fit.md) | AdExp f-I vs 在线 ODE vs LIF，NFR-4 |
| 在线 vs 降阶 | [M1_online_vs_reduced.md](M1_online_vs_reduced.md) | RMSE / 墙钟 / 验收 JSON 中的 VP |
| chirp \|Z(ω)\| | [M1_chirp_impedance.md](M1_chirp_impedance.md) | 频域峰值与层级判定 |

```bash
uv run --no-sync nrde fetch fly --tier pretrained
uv run --no-sync python scripts/make_figures.py --verify
```

重算图与 ledger：

```bash
uv run --no-sync python scripts/make_figures.py
```
