# M1 — 在线 ODE vs 降阶 F

**状态**：精度用默认数据集 f-I 留出重算；墙钟见证据链（主机相关）  
**对应**：RFC-001 验收条 2；NFR-4 / 回路墙钟（NFR-5 的 VP 仍以 `datasets/fly/reports/acceptance.json` 为准）

## 协议

单胞 f-I：见 [M1_adexp_fit.md](M1_adexp_fit.md)。  
回路墙钟：MaleCNS 附属屈肌运动神经元回路，249 节点、4622 边、200 步、\(I_\mathrm{ext}=0.4\) nA。ODE 墙钟由 8 细胞 × 200 ms 线性外推到 249。

## 结果摘要

| 模型 / 方法 | 层级 | 相对 AdExp ODE | 249×200 ms 墙钟 |
|---|---|---|---|
| 在线 AdExp ODE | — | RMSE 0 | ~1.8 s |
| NRDE \(F\)（aCC） | meta L1 | \(R^2=0.995\)，RMSE 1.6 Hz | ~15 ms |
| 统一 LIF | L0 | RMSE 110 Hz | ~12 ms |
| aCC L1 VP（验收 JSON） | L1 | 相对 VP 0.004 | — |

NFR-5 的 Victor–Purpura 在导出数据集时写入 `datasets/fly/reports/acceptance.json`（`aCC_l1_relative_vp`），本页不另造一套脉冲协议。

## 复现

```bash
uv run --no-sync python scripts/make_figures.py --verify
uv run --extra dev pytest tests/test_validation.py -q
```
