# 从源码构建与完整开发环境

面向贡献者与需要本地复现全部产品面（A1 / A2 / B / C）的开发者。  
**终端用户**请先看仓库 [README 安装矩阵](https://github.com/NACXA0/NeuronReducedDynamicsEngine#安装矩阵终稿四行)；本页是「装齐一切」的清单。

## 1. 最低 vs 完整

| 目标 | 需要什么 | 不需要什么 |
|---|---|---|
| 跑核心引擎 + 拟合（B/C） | Python + 可编辑安装 | FlyGym、Julia、连接组 Feather |
| 贡献者默认（CI 主 lane） | + `[dev]` | 同上 |
| A2 真 FlyGym 闭环 | + `[fly]`，Python **3.12–3.14** | Julia |
| A1 真 GIF | + `[demo-assets]`（imageio） | 真 MuJoCo（Mock 亦可出占位/简易 GIF） |
| C 层 PySR 精化 | + `[pysr]` + 系统 Julia | FlyGym |
| 全量 MaleCNS 回路标定 | `uv run nrde fetch fly --tier full` 或下载脚本 | 不必装进 wheel |

核心原则（NFR-11 / RFC-002）：**主包安装不得因 Julia / FlyGym 失败**；可选能力一律 extras + guarded import。

## 2. 系统与工具链

| 项 | 要求 | 说明 |
|---|---|---|
| OS | Linux / macOS（Windows 未作为 CI 目标） | 连接组下载与 FlyGym 在 Linux 上最省心 |
| Git | 任意近期版本 | 克隆本仓 |
| Python | **≥ 3.10**；贡献推荐 **3.11–3.14** | CI 主 lane **3.11**；3.14 为 allow-failure |
| 包管理 | [`uv`](https://github.com/astral-sh/uv) | 只写入仓库下的 `.venv`；不要 `pip install` 进系统 Python |
| 可选：Julia | 仅 `[pysr]` | 由 PySR 拉取/绑定；主包不捆绑 |
| 可选：MuJoCo / 显示 | 仅真 FlyGym 可视化 | A2 `--headless` 与 A1 Mock **不**依赖 X server |
| 可选：ffmpeg | 仅真机位视频导出 | 当前 `render_demo.py` 默认 headless 帧/GIF，不强制 |

无 GPU 要求。JAX / Numba / Torch 均为 extras，**核心路径不依赖**。在线 GPU 稀疏走 `nrde[torch]`（cuSPARSE），不是 JAX sparse。

## 3. 克隆与安装

安装只做这一步。后面的运行、测试各自成块，都用 `uv run`。

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine
uv sync --extra dev
```

运行：

```bash
uv run nrde --help
uv run python -c "import nrde; print(nrde.__version__, nrde.__all__)"
```

测试：

```bash
uv run --extra dev pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
uv run --extra dev ruff check src tests examples scripts
```

可选 pre-commit（不进默认 extra）：

```bash
uv run --with pre-commit pre-commit install
```

## 4. Python extras 全表（`pyproject.toml`）

| Extra | 安装 | 拉取的库 | 用途 | Python 注意 |
|---|---|---|---|---|
| （无） | `uv sync` | numpy、scipy、pyyaml、pyarrow | B 引擎 + C 管线 | ≥3.10 |
| `dev` | `uv sync --extra dev` | + pytest、pytest-cov、ruff | 贡献 / CI 主 lane | — |
| `fly` / `flygym` | `uv sync --extra fly` | + flygym≥2.1 | A2 真具身；缺省时 Mock | **仅 3.12≤py&lt;3.15** 声明安装；3.11 上 Mock 仍可跑 demo |
| `pysr` | `uv sync --extra pysr` | + pysr | C 可选符号回归精化 | 需系统 Julia；失败则退场 |
| `docs` | `uv sync --extra docs` | + mkdocs、mkdocs-material | 本地文档站 | `uv run mkdocs serve` |
| `demo-assets` | `uv sync --extra demo-assets` | + imageio（及 pillow） | A1 `render_demo` 写出 GIF | 无则写 NPZ/TXT 占位 |
| `jax` | `uv sync --extra jax` | + jax | 离线 f-I `vmap`/`scan`；**非** GPU 稀疏主路径 | 可选 |
| `numba` | `uv sync --extra numba` | + numba | 离线 Euler/LUT 可选 JIT（AVX-512 由 LLVM 发出）；缺省走同一标量核 | 可选；3.14 可能尚未支持 |
| `torch` | 不要 `uv sync --extra torch` 来选设备 | + torch | 在线 CSR SpMV；CUDA 走 cuSPARSE | 见 [稀疏传播](quickstart/engine.md)：CPU / CUDA 索引分开装；已有 torch 时 `uv sync --inexact`，运行加 `--no-sync` |

一次装齐「开发 + 文档 + GIF + 果蝇」（在支持的 Python 上）。PySR 另开一行，避免没有 Julia 的环境被拖垮。

```bash
uv sync --extra dev --extra docs --extra demo-assets --extra fly
```

```bash
uv sync --extra pysr
```

## 5. 数据与产物（按体积）

| 层级 | 内容 | 如何得到 | 体积量级 | 是否入库 |
|---|---|---|---|---|
| 种子 artifacts | `artifacts/type_0.npz`、`aCC.npz`、`hh_demo.npz`（+ meta） | 克隆仓库；`uv run nrde fetch fly --tier smoke\|seed` 校验 | MB | 是（类型键种子） |
| Demo 配置 | `configs/flygym_demo.yaml`、`configs/manifests/fly.yaml` | 克隆 | KB | 是 |
| MaleCNS 注释+递质 | Feather 两件 | `uv run nrde fetch fly --tier full` 或 `uv run python scripts/download_malecns.py --skip-weights` | ≈55 MB | 否（`data/` gitignore） |
| MaleCNS 权重 | `connectome-weights-*.feather` | 同上去掉 `--skip-weights` | ≈1.1 GB | 否 |
| FlyWire 对照 | 自备导出 | 见 `data/README.md` | 不定 | 否；**禁止与 MaleCNS 混用 ID** |
| A1 离线包 | `dist/nrde-fly-offline.zip` | `uv run python scripts/make_release.py` | MB | 否（gitignore） |
| 电生理拟合原料 | ModelDB / NeuroElectro / 文献 | 研究自备；**不**由连接组产生 f-I | — | 不随主包 |

常用命令：

常用命令。`smoke` 只校验种子哈希，`seed` 用仓内种子，`full` 再拉 MaleCNS。

```bash
uv run nrde fetch fly --tier smoke
uv run nrde fetch fly --tier seed
uv run nrde fetch fly --tier full
```

```bash
uv run python scripts/download_malecns.py --skip-weights
uv run python scripts/download_malecns.py
```

路径约定：`data/malecns/`、`data/flywire/` — 详见 [`data/README.md`](../data/README.md) 与 [数据源](data_sources.md)。

> **注意**：`uv run python examples/01_single_neuron_fit.py` 会写入 `artifacts/aCC.npz`。覆写后 `uv run nrde fetch` 可能因 sha256 / `config_hash` 失配失败。改种子后请同步更新 `configs/manifests/fly.yaml` 与 `configs/flygym_demo.yaml` 中的哈希，或从已知良好副本恢复。

## 6. 按产品面的「装什么、跑什么」

### B / C（引擎与管线）— 默认开发

安装：

```bash
uv sync --extra dev
```

运行：

```bash
uv run nrde offline fit --model adexp --type-id aCC_local --lut --chirp --out /tmp/nrde_fit/aCC_local.npz
uv run nrde offline validate --fit /tmp/nrde_fit/aCC_local.npz
uv run nrde run simulate --fit /tmp/nrde_fit/aCC_local.npz --n 80 --steps 50
uv run python examples/04_custom_model_pipeline.py
```

怎么从零拟合，见 [拟合 F](quickstart/pipeline.md)。拟合之后的稀疏传播，见 [稀疏传播](quickstart/engine.md)。不要把结果写回 `artifacts/aCC.npz`。

### A2（果蝇预设）

安装：

```bash
uv sync --extra dev --extra fly
```

运行：

```bash
uv run nrde fetch fly --tier seed
uv run nrde demo flygym --steps 100 --headless
```

### A1（体验包 / Demo 卡）

安装：

```bash
uv sync --extra dev --extra demo-assets
```

运行：

```bash
uv run python scripts/render_demo.py --preset flygym-demo-v01 --steps 40 --seed 0 --out assets/demo/
uv run python scripts/make_colab.py --out assets/demo/demo.ipynb
uv run python scripts/make_release.py --out dist/nrde-fly-offline.zip
```

细节：[A1 资产管线](asset_pipeline.md)。

### 文档站

安装：

```bash
uv sync --extra docs
```

运行：

```bash
uv run mkdocs serve
```

### PySR（可选，默认可退场）

先按 PySR 文档安装 Julia。

安装：

```bash
uv sync --extra pysr
```

运行：

```bash
uv run nrde offline validate --pysr-exam
```

测试：

```bash
uv run --extra pysr pytest -m pysr
```

准入政策：[PySR 准入](pysr_admission.md)。

## 7. 测试矩阵（与 CI 对齐）

| Lane | 命令 | CI |
|---|---|---|
| 裸装 smoke | `uv sync` 后 `uv run nrde demo` / `uv run nrde fetch fly --tier smoke` | `ci.yml` bare |
| 核心 | `uv run --extra dev pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"` | `ci.yml` test |
| FlyGym | `uv run --extra fly pytest -m flygym` | `ci-flygym.yml` |
| PySR | `uv run --extra pysr pytest -m pysr` | `ci-pysr.yml`（可失败） |
| A1 资产 | 见 `ci-demo-assets.yml` | weekly / 手工 |

## 8. 「全部开发」一键清单（核对用）

- [ ] Python 3.11+（真 FlyGym 用 3.12–3.14）
- [ ] `uv sync --extra dev`
- [ ] （可选）`uv run --with pre-commit pre-commit install`
- [ ] （可选）再 `uv sync` 加上 `--extra fly` / `docs` / `demo-assets` / `pysr`
- [ ] 种子：`uv run nrde fetch fly --tier seed`
- [ ] （可选）MaleCNS：`uv run nrde fetch fly --tier full` 或 `uv run python scripts/download_malecns.py`
- [ ] `uv run --extra dev pytest` 与 `uv run --extra dev ruff check` 通过
- [ ] （可选）A1 三脚本产出 GIF / ipynb / zip

## 9. 相关文档

- 贡献流程：[CONTRIBUTING.md](../CONTRIBUTING.md)
- 数据源 / fetch：[data_sources.md](data_sources.md)
- A1 管线：[asset_pipeline.md](asset_pipeline.md)
- 架构与 extras 决策：[RFC-002](rfcs/RFC-002.md)
