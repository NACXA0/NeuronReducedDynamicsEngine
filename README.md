# NRDE — Neuron Reduced-Dynamics Engine

连接组进，按细胞类型绑定的降阶函数 \(F\) 出。运行时做稀疏传播再查 \(F\)，不再逐步积分 AdExp / HH。果蝇 MaleCNS 是当前实例，不是引擎的全部范围。

## 介绍

一边是 FlyBrainLab 这类逐步解单胞 ODE 的仿真：保真，全脑太贵。另一边是全脑统一 LIF：只要连接表加一条曲线，快，但所有细胞共用同一个函数。NRDE 走中间：输入仍是连接表和类型；离线把每一类的源 ODE 拟合成 \(F\)；在线只做

\[
I = W\,(r\,\Delta t) + I_{\mathrm{ext}},\qquad r = F(I).
\]

主线里 \(F\) 只按**类型**绑定。脑区、坐标、调质没有可拟合的数据，不改变已发布表，也不需要重新训练。它们作为可插拔管道步留给深度开发，也可以把 \(F\) 换成与 ODE 无关的映射，用来试抽象规则。这不是宣称已经重建真实电生理，也不主张虚拟果蝇能走出有意义的行为。

口径以 [RFC-001](docs/rfcs/RFC-001.md) 为准。初衷全文在 [初始设想](docs/初始设想.md)。

发布集在 [`datasets/fly/`](datasets/fly/README.md)：aCC / T4 共用 AdExp 替身，默认兴奋是 ExpLIF，默认抑制是 LIF，另有 249 节点、4622 边的附属屈肌运动神经元回路。

## 宣传

相对在线 AdExp（同一电流网格的留出中点），NRDE 的发放率 RMSE 为 **1.61 Hz**（\(R^2=0.994607\)），249 细胞 × 200 ms 约 **122×** 快于逐步 Euler。同一墙上统一 LIF 只要 12 ms，但相对 AdExp 偏出 **110.42 Hz**。

| 方法 | RMSE (Hz) | \(R^2\) | 249 × 200 ms |
|---|---:|---:|---:|
| 在线 AdExp ODE | 0 | 1 | 1805.8 ms |
| NRDE \(F\) | 1.61 | 0.994607 | 14.83 ms |
| 统一 LIF | 110.42 | −24.25 | 11.98 ms |

主机：AMD Ryzen 7 5800X，NumPy。墙钟换机器会变。相对的是本库 AdExp Euler，不是活体、也不是 FlyBrainLab。

![精度 vs 代价](docs/reports/figures/hero_pareto.svg)

同一组默认表上的验收（`datasets/fly/reports/acceptance.json`）：

| 项 | 结果 |
|---|---|
| NFR-4 单胞 f–I | 通过。留出 \(R^2=0.994607\)（门限 ≥ 0.98） |
| NFR-5 脉冲 VP | 通过。aCC L1 相对距离 0.00395，LIF L1 为 0.064（门限 ≤ 0.10） |
| Shiu α | 通过。持出倍率 3 收回 2.99986，相对发放误差 \(5.7\times10^{-5}\) |
| 具身接口 | 通过。headless 为 Mock，≥100 步 |
| 覆盖率 | 0.7745（门限 0.70）。PySR 未装，记为 retired |
| L3 / FNO | 未做。记录 `deferred_v0.2`，没有伪造表 |

协议、曲线、哈希和重算命令在 [证据链](docs/reports/evidence.md)。

## 快速入门

不要装进系统 Python。仓库下用 `uv`，命令一律 `uv run`。

安装：

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine
uv sync
```

运行（表已拟合，这条命令只做稀疏传播）：

```bash
uv run nrde fetch fly --tier pretrained
uv run nrde run simulate \
  --fit datasets/fly/fits/aCC_l0.npz \
  --n 80 --steps 50
```

测试：

```bash
uv run --extra dev pytest -m "not pysr and not flygym"
```

克隆后 `fetch --tier pretrained` 只校验仓库里 33 个发布 npz 的 sha256，标准错误应是「已校验，未下载」。工作区里没有这些表时，才会按清单里的地址下载。GPU、全量 MaleCNS、自己拟合，见下面的深度文档。

## 文档

细节不在本页。入口地图也在 [docs/index.md](docs/index.md)。

### 使用

| 文档 | 内容 |
|---|---|
| [果蝇 Demo](docs/quickstart/demo.md) | Colab / GIF，`nrde demo flygym` |
| [拟合 F](docs/quickstart/pipeline.md) | L0 / L1 / L2、自动升层、效率/中性/拟真 |
| [稀疏传播](docs/quickstart/engine.md) | `nrde run simulate`，CPU / CUDA |
| [从源码构建](docs/from_source.md) | extras、Julia、FlyGym、贡献者环境 |

### 深度开发

| 文档 | 内容 |
|---|---|
| [自定义管道](docs/quickstart/extend.md) | 追加 RateOp、换掉类型表、系数改成函数 |
| [ModelSpec](docs/model_spec_protocol.md) | 自有 ODE 怎么进拟合 |
| [API](docs/api.md) | 顶层五符号与子模块 |
| [不适用清单](docs/known-inapproximable.md) | 降阶失败时的在线 ODE 回退 |

### 数据与证据

| 文档 | 内容 |
|---|---|
| [默认数据集](datasets/fly/README.md) | `fits/`、`precision/`、`sparse/`、`prefer/` |
| [证据链](docs/reports/evidence.md) | 上表数字的协议与复现 |
| [验证报告](docs/reports/README.md) | M1 拟合 / 在线 vs 降阶 / chirp |
| [数据源](docs/data_sources.md) | `fetch` 的 smoke / seed / pretrained / prefer / full |

### 规格

冲突以 RFC 为准。中文规格是历史快照。

| 文档 | 内容 |
|---|---|
| [RFC-001](docs/rfcs/RFC-001.md) | 拟合分层、宣传口径 |
| [RFC-002](docs/rfcs/RFC-002.md) | 包结构与产品面 |
| [需求文档](docs/需求文档.md) · [计划书](docs/项目计划书.md) · [设计说明书](docs/系统设计说明书.md) | FR / 里程碑 / 两阶段架构 |
| [贡献约定](CONTRIBUTING.md) · [Changelog](docs/CHANGELOG.md) | 命名、测试、版本。发布与数据维护在 [maintainer](maintainer/README.md) |

许可证：[Apache License 2.0](LICENSE)。
