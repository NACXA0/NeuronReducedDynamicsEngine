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
| 全量 MaleCNS 回路标定 | `nrde fetch fly --tier full` 或下载脚本 | 不必装进 wheel |

核心原则（NFR-11 / RFC-002）：**主包安装不得因 Julia / FlyGym 失败**；可选能力一律 extras + guarded import。

## 2. 系统与工具链

| 项 | 要求 | 说明 |
|---|---|---|
| OS | Linux / macOS（Windows 未作为 CI 目标） | 连接组下载与 FlyGym 在 Linux 上最省心 |
| Git | 任意近期版本 | 克隆本仓 |
| Python | **≥ 3.10**；贡献推荐 **3.11–3.14** | CI 主 lane **3.11**；3.14 为 allow-failure |
| 包管理 | `pip` 或 [`uv`](https://github.com/astral-sh/uv) | 下文两种写法等价 |
| 可选：Julia | 仅 `[pysr]` | 由 PySR 拉取/绑定；主包不捆绑 |
| 可选：MuJoCo / 显示 | 仅真 FlyGym 可视化 | A2 `--headless` 与 A1 Mock **不**依赖 X server |
| 可选：ffmpeg | 仅真机位视频导出 | 当前 `render_demo.py` 默认 headless 帧/GIF，不强制 |

无 GPU 要求。JAX 为实验性 extra，**核心路径不依赖**。

## 3. 克隆与可编辑安装

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine

# 推荐：贡献者默认（B/C + 测试/lint）
pip install -e ".[dev]"
# 或
uv sync --extra dev
```

验证：

```bash
nrde --help
python -c "import nrde; print(nrde.__version__, nrde.__all__)"
pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
ruff check src tests examples scripts
```

可选 pre-commit（仓库已含配置）：

```bash
pip install pre-commit
pre-commit install
```

## 4. Python extras 全表（`pyproject.toml`）

| Extra | 安装命令 | 拉取的库 | 用途 | Python 注意 |
|---|---|---|---|---|
| （无） | `pip install -e .` | numpy、scipy、pyyaml、pyarrow | B 引擎 + C 管线 | ≥3.10 |
| `dev` | `".[dev]"` | + pytest、pytest-cov、ruff | 贡献 / CI 主 lane | — |
| `fly` / `flygym` | `".[fly]"` 或 `".[flygym]"` | + flygym≥2.1 | A2 真具身；缺省时 Mock | **仅 3.12≤py&lt;3.15** 声明安装；3.11 上 Mock 仍可跑 demo |
| `pysr` | `".[pysr]"` | + pysr | C 可选符号回归精化 | 需系统 Julia；失败则退场 |
| `docs` | `".[docs]"` | + mkdocs、mkdocs-material | 本地文档站 | `mkdocs serve` |
| `demo-assets` | `".[demo-assets]"` | + imageio（及 pillow） | A1 `render_demo` 写出 GIF | 无则写 NPZ/TXT 占位 |
| `jax` | `".[jax]"` | + jax | 实验加速；非默认路径 | 可选 |

一次装齐「开发 + 文档 + GIF + 果蝇」（在支持的 Python 上）：

```bash
pip install -e ".[dev,docs,demo-assets,fly]"
# PySR 另开一行，避免拖垮无 Julia 的环境：
# pip install -e ".[pysr]"
```

`uv` 等价：

```bash
uv sync --extra dev --extra docs --extra demo-assets --extra fly
```

## 5. 数据与产物（按体积）

| 层级 | 内容 | 如何得到 | 体积量级 | 是否入库 |
|---|---|---|---|---|
| 种子 artifacts | `artifacts/type_0.npz`、`aCC.npz`、`hh_demo.npz`（+ meta） | 克隆仓库；`nrde fetch fly --tier smoke\|seed` 校验 | MB | 是（类型键种子） |
| Demo 配置 | `configs/flygym_demo.yaml`、`configs/manifests/fly.yaml` | 克隆 | KB | 是 |
| MaleCNS 注释+递质 | Feather 两件 | `nrde fetch fly --tier full` 或 `python scripts/download_malecns.py --skip-weights` | ≈55 MB | 否（`data/` gitignore） |
| MaleCNS 权重 | `connectome-weights-*.feather` | 同上去掉 `--skip-weights` | ≈1.1 GB | 否 |
| FlyWire 对照 | 自备导出 | 见 `data/README.md` | 不定 | 否；**禁止与 MaleCNS 混用 ID** |
| A1 离线包 | `dist/nrde-fly-offline.zip` | `python scripts/make_release.py` | MB | 否（gitignore） |
| 电生理拟合原料 | ModelDB / NeuroElectro / 文献 | 研究自备；**不**由连接组产生 f-I | — | 不随主包 |

常用命令：

```bash
nrde fetch fly --tier smoke    # 只校验种子 sha256
nrde fetch fly --tier seed     # 种子（已在仓内则可离线）
nrde fetch fly --tier full     # 种子 + MaleCNS（可再 --refresh）
python scripts/download_malecns.py --skip-weights   # 小文件起步
python scripts/download_malecns.py                  # 含 ~1.1 GB 权重
```

路径约定：`data/malecns/`、`data/flywire/` — 详见 [`data/README.md`](../data/README.md) 与 [数据源](data_sources.md)。

> **注意**：`examples/01_single_neuron_fit.py` 会写入 `artifacts/aCC.npz`。覆写后 `nrde fetch` 可能因 sha256 / `config_hash` 失配失败。改种子后请同步更新 `configs/manifests/fly.yaml` 与 `configs/flygym_demo.yaml` 中的哈希，或从已知良好副本恢复。

## 6. 按产品面的「装什么、跑什么」

### B / C（引擎与管线）— 默认开发

```bash
pip install -e ".[dev]"
nrde offline fit --model adexp --type-id aCC
nrde run simulate --fit artifacts/aCC.npz --n 80 --steps 50
python examples/04_custom_model_pipeline.py
```

### A2（果蝇预设）

```bash
pip install -e ".[dev,fly]"          # 无 flygym wheel 时仍可用 Mock
nrde fetch fly --tier seed
nrde demo flygym --steps 100 --headless
```

### A1（体验包 / Demo 卡）

```bash
pip install -e ".[dev,demo-assets]"
python scripts/render_demo.py --preset flygym-demo-v01 --steps 40 --seed 0 --out assets/demo/
python scripts/make_colab.py --out assets/demo/demo.ipynb
python scripts/make_release.py --out dist/nrde-fly-offline.zip
```

细节：[A1 资产管线](asset_pipeline.md)。

### 文档站

```bash
pip install -e ".[docs]"
mkdocs serve
```

### PySR（可选，默认可退场）

```bash
# 先按 PySR 文档安装 Julia，再：
pip install -e ".[pysr]"
nrde offline validate --pysr-exam
pytest -m pysr
```

准入政策：[PySR 准入](pysr_admission.md)。

## 7. 测试矩阵（与 CI 对齐）

| Lane | 命令 | CI |
|---|---|---|
| 裸装 smoke | `pip install -e .` 后 `nrde demo` / `fetch --tier smoke` | `ci.yml` bare |
| 核心 | `pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"` | `ci.yml` test |
| FlyGym | `pytest -m flygym` | `ci-flygym.yml` |
| PySR | `pytest -m pysr` | `ci-pysr.yml`（可失败） |
| A1 资产 | 见 `ci-demo-assets.yml` | weekly / 手工 |

## 8. 「全部开发」一键清单（核对用）

- [ ] Python 3.11+（真 FlyGym 用 3.12–3.14）
- [ ] `pip install -e ".[dev]"`（或 `uv sync --extra dev`）
- [ ] （可选）`pre-commit install`
- [ ] （可选）`".[fly]"` / `".[docs]"` / `".[demo-assets]"` / `".[pysr]"`
- [ ] 种子：`nrde fetch fly --tier seed`
- [ ] （可选）MaleCNS：`fetch --tier full` 或 `download_malecns.py`
- [ ] 核心测试与 ruff 通过
- [ ] （可选）A1 三脚本产出 GIF / ipynb / zip

## 9. 相关文档

- 贡献流程：[CONTRIBUTING.md](../CONTRIBUTING.md)
- 数据源 / fetch：[data_sources.md](data_sources.md)
- A1 管线：[asset_pipeline.md](asset_pipeline.md)
- 架构与 extras 决策：[RFC-002](rfcs/RFC-002.md)
