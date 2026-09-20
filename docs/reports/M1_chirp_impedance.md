# M1 — Chirp \|Z(ω)\|

**状态**：已用默认 AdExp / LIF 重算  
**对应**：RFC-001 FR-3.6 / FR-3.9；[证据链](evidence.md)

## 协议

- 亚阈值正弦扫频，锁相估计 \(\lvert Z(\omega)\rvert = \lvert V_{ac}\rvert / \lvert I_{ac}\rvert\)
- 内点峰值 → 建议升至 **L2**（SRM / 传递函数族）
- 无峰且无慢变量 → 保持 L0；AdExp 无峰仍可因适应升 L1

## 结果摘要

| 模型 | 有峰 | 峰频 (Hz) | 层级建议 |
|---|---|---|---|
| AdExp（aCC） | 否 | — | L1（适应 / 状态 LUT） |
| LIF（默认抑制） | 否 | — | L0 |

`aCC_l0.npz` 内存储的 \(|Z|\) 与重算曲线同形（2 Hz 处略高、随后滚降）。判定原文：无内点峰。图：`figures/adexp_lif_chirp.svg`。

```bash
uv run python scripts/make_figures.py
```
