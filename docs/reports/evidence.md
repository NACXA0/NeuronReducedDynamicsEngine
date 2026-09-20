# 宣传数字的证据链

README 上的 \(R^2\)、RMSE 和墙钟都从默认数据集 **当场重算**，不是手填。本页说明数字从哪来、怎么核验、什么不能从这组图推出去。

**需要单独成文。** README 只引用结果；协议、输入哈希、主机和复现命令放在这里和 `figures/evidence.json`。否则版本一变，图上的数会变成无法审计的截图。

## 核验（不重算）

```bash
uv run --no-sync nrde fetch fly --tier pretrained
uv run --no-sync python scripts/make_figures.py --verify
```

第一条核对 `datasets/fly/` 里 **33** 个预训练 `.npz` 的 sha256（与 `configs/manifests/fly.yaml` 的 `pretrained` 档一致）。第二条核对本目录 CSV/SVG 的 sha256 是否仍等于 `figures/evidence.json` 里登记的值。

## 重算（会改墙钟，会改输出哈希）

```bash
uv run --no-sync python scripts/make_figures.py
```

精度（f-I \(R^2\) / RMSE / chirp）在同一协议下应可复现。墙钟随 CPU 变化，ledger 里写了主机；README 引用数量级和 CPU 型号，不把毫秒写死成跨机器承诺。

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

负的 LIF \(R^2\) 不是算错：默认 LIF（\(C=0.1\) nF、无适应）在同一 \(I\) 上发放远快于 AdExp。README 用 RMSE（Hz）报这件事，避免把负 \(R^2\) 读成程序故障。

## 本次锁定的结果

以仓库内 `figures/evidence.json` 为准（生成时主机：AMD Ryzen 7 5800X，CPython 3.14，NumPy 引擎）：

| 方法 | 相对 AdExp ODE 的 \(R^2\) | RMSE (Hz) | 249×200 ms 墙钟 |
|---|---|---|---|
| 在线 AdExp ODE | 1（定义） | 0 | ~1.8 s（8 细胞外推） |
| NRDE \(F\)（aCC 表） | **0.995** | **1.6** | ~15 ms |
| 统一 LIF | 负（曲线形状/增益都不贴） | **110** | ~12 ms |

NFR-4（\(R^2 \ge 0.98\)）通过。拟合器写入 meta 的 \(R^2=0.9947\) 与留出中点重算的 \(0.9946\) 一致。

## 口径

允许：类型可绑定不同 \(F\)；降阶 \(F\) 在 AdExp 单胞上贴近在线 ODE，运行时接近统一 LIF 的查表代价。

禁止从本页推出：已重建真实 aCC/T4 电生理；虚拟果蝇有意义行走；ODE 墙钟未外推；跨 MaleCNS / FlyWire ID 的网络级对照。
