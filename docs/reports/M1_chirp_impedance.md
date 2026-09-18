# M1 — Chirp \|Z(ω)\|

**状态**：骨架  
**对应**：RFC-001 FR-3.6 / FR-3.9；`nrde.fitting.chirp`

## 协议

- 亚阈值正弦扫频，锁相估计 \(\lvert Z(\omega)\rvert = \lvert V_{ac}\rvert / \lvert I_{ac}\rvert\)
- 内点峰值 → 建议升至 **L2**（SRM / 传递函数族）
- 无峰且无慢变量 → 保持 L0

## 结果摘要

| 模型 | 有峰 | 峰频 (Hz) | 层级建议 |
|---|---|---|---|
| AdExp | （填） | | |
| ExpLIF | | | |
| HH | | | |

```bash
python scripts/make_figures.py --models adexp explif hh
```

## 图

- `figures/M1_*_chirp.csv` — `freq_Hz,Z_abs`
- `figures/M1_*_chirp_note.txt` — 峰值判定原文
