# 宣传数字的证据链

README 只放对比表、验收摘要和 Pareto 图。曲线、摘录、协议、哈希和复现命令在本页。数字从默认数据集当场重算，不是手填。登记文件是 `figures/evidence.json`。

## 核验（不重算）

```bash
uv run nrde fetch fly --tier pretrained
uv run python scripts/make_figures.py --verify
```

第一条核对 `datasets/fly/` 里 **33** 个预训练 `.npz` 的 sha256（与 `configs/manifests/fly.yaml` 的 `pretrained` 档一致）。第二条核对本目录 CSV/SVG 的 sha256 是否仍等于 `figures/evidence.json` 里登记的值。

## 重算（会改墙钟，会改输出哈希）

```bash
uv run python scripts/make_figures.py
```

精度（f-I \(R^2\) / RMSE / chirp）在同一协议下应可复现。墙钟随 CPU 变化。改完后把 README 对比表里的锁定数字一并更新；`tests/test_evidence_ledger.py` 会检查 README 是否仍引用账本中的 \(R^2\)、RMSE 和墙钟。

## 输入

| 角色 | 路径 | 说明 |
|---|---|---|
| Ground truth 细胞 | `datasets/fly/fits/aCC_l0.npz` | AdExp 稳态 f-I。文件名是 L0，拟合器因适应把 `layer` 标成 **L1** |
| 统一 LIF | `datasets/fly/fits/default_inhibitory_l0.npz` | 默认抑制细胞；真正的一维 f-I |
| 回路 | `datasets/fly/sparse/accessory_mn_circuit.feather` | 69 个附属屈肌运动神经元 + 180 个最强上游 = **249** 点、**4622** 边 |
| 类型键 | 同目录 `.meta.json` | T4 与 aCC 共用 AdExp 替身表，只改 type_id |

全量 MaleCNS Feather（约 1.1 GB）仍在 `data/malecns/`，清单只登记 sha256，不复制进 `datasets/fly/`。L3 / FNO 见 `datasets/fly/reports/l3.json`：v0.2，没有造假表。

## 协议

相对 **在线 AdExp Euler**（`dt = 0.05` ms），不是相对实验细胞，也不是相对 FlyBrainLab 全脑 HH。

| 项 | 设定 |
|---|---|
| f-I 电流 | `aCC_l0` 训练网格的 **中点**（相对结点为留出） |
| 积分 | `t_total = 800` ms，窗口 `400` ms，与导出脚本相同 |
| 墙钟回路 | 249 节点、200 步、引擎 `dt = 1` ms、\(I_\mathrm{ext}=0.4\) nA |
| ODE 墙钟 | 8 个 AdExp 细胞 × 200 ms Euler，**线性外推到 249** |
| NRDE \(F\) | 屈肌运动神经元绑 aCC 表，其余绑默认兴奋（ExpLIF）表 |
| 统一 LIF | 全图绑 `default_inhibitory_l0` |

负的 LIF \(R^2\) 不是算错：默认 LIF（\(C=0.1\) nF、无适应）在同一 \(I\) 上发放远快于 AdExp。

## 本次锁定的结果

主机：AMD Ryzen 7 5800X，CPython 3.14，NumPy。以 `figures/evidence.json` 为准。

| 方法 | 相对 AdExp ODE 的 \(R^2\) | RMSE (Hz) | 249×200 ms 墙钟 |
|---|---:|---:|---:|
| 在线 AdExp ODE | 1 | 0 | 1805.8 ms |
| NRDE \(F\)（aCC 表） | 0.994607 | 1.61 | 14.83 ms |
| 统一 LIF | −24.25 | 110.42 | 11.98 ms |

拟合器写入 meta 的 \(R^2\) 是 0.994698。上表是留出中点重算。NFR-4（≥ 0.98）通过。

同一回路 200 步后的平均发放率：按类型绑定 \(F\) 为 133.8 Hz，全图改绑统一 LIF 为 219.5 Hz。这是两条策略的输出差，不是相对 ODE 的误差。

脉冲时刻见 `datasets/fly/reports/acceptance.json`（1 s 窗口，Victor–Purpura 相对距离，门限 ≤ 0.10）：aCC 的 L1 表为 0.00395，默认 LIF 的 L1 表为 0.0640。

![精度 vs 代价](figures/hero_pareto.svg)

### f–I

完整 15 点：[adexp_fi_overlay.csv](figures/adexp_fi_overlay.csv)。

| \(I\) (nA) | AdExp ODE (Hz) | NRDE \(F\) | 统一 LIF |
|---:|---:|---:|---:|
| 0.187 | 0 | 0 | 0 |
| 0.240 | 5.0 | 0 | 57.5 |
| 0.400 | 22.5 | 23.8 | 132.5 |
| 0.560 | 40.0 | 41.3 | 182.5 |
| 0.773 | 62.5 | 62.5 | 230.0 |

![AdExp f–I](figures/adexp_fi_overlay.svg)

### 阻抗谱

两组都没有内点谐振峰。AdExp 仍因适应被标成 L1。完整 7 点：[adexp_lif_chirp.csv](figures/adexp_lif_chirp.csv)。

| 频率 (Hz) | AdExp \|Z\| | LIF \|Z\| |
|---:|---:|---:|
| 2 | 91.7 | 98.9 |
| 10 | 64.6 | 86.7 |
| 80 | 16.9 | 22.8 |

![亚阈值阻抗谱](figures/adexp_lif_chirp.svg)

## 口径

允许：类型可绑定不同 \(F\)；降阶 \(F\) 在 AdExp 单胞上贴近在线 ODE，运行时接近统一 LIF 的查表代价。

禁止从本页推出：已重建真实 aCC/T4 电生理；虚拟果蝇有意义行走；ODE 墙钟未外推；跨 MaleCNS / FlyWire ID 的网络级对照。
