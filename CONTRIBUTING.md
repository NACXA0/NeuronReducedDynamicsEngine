# 贡献指南

感谢参与 NRDE（Neuron Reduced-Dynamics Engine）开发。

## 命名约定

| 角色 | 名称 |
|---|---|
| Git 仓库 | `NeuronReducedDynamicsEngine` |
| 宣传全称 | Neuron Reduced-Dynamics Engine（NRDE） |
| 中文全称 | 神经元降阶动力学引擎 |
| PyPI / 发行名 | `neuron-reduced-dynamics-engine` |
| Python import / CLI | `nrde` |
| 方法术语 | state approximation / 状态近似 |

三者可以不同，但请勿再引入第四个品牌名。本地 `*.egg-info` 由 setuptools 生成，**不要提交**。

## 环境

- 推荐 Python **3.11–3.14**（开发可跟进最新稳定版；CI 主 lane 使用 3.11）。
- 安装：`uv sync --extra dev`
- 可选再加：`--extra pysr`、`--extra fly`、`--extra docs`、`--extra demo-assets`

**完整清单**（系统依赖、全部 extras、种子 / MaleCNS 数据、A1 构建）：见 [docs/from_source.md](docs/from_source.md)。请勿把大体积 Feather 提交进 Git。

公共 import 面仅五符号：`make` / `demo` / `run` / `offline` / `ModelSpec`（RFC-002）。深度 API 从子模块导入。

产品面：**A1** 奇观（Colab/GIF/Release）· **A2** 预设（`nrde[fly]` + `nrde fetch`）· **B** 引擎 · **C** 管线。A1 资产脚本见 `docs/asset_pipeline.md`。

## 开发流程

安装：

```bash
uv sync --extra dev
```

测试：

```bash
uv run --extra dev ruff check src tests examples
uv run --extra dev pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
```

- 核心测试默认不装 FlyGym / Julia。
- `@pytest.mark.flygym` / `pysr` / `slow` 由独立 CI lane 或本地显式运行。
- 连接组原始数据放在 `data/`（已 gitignore），用 `scripts/download_malecns.py` 拉取。

## 文档

- 中文规格文件名保持中文；内容可中英混排。
- 验证报告写在 `docs/reports/`，并加入 `mkdocs.yml` 导航。
- 行为变更请更新 `docs/CHANGELOG.md`；架构级变更走 `docs/rfcs/`。

## PR 检查清单

- [ ] 测试与 ruff 通过（或说明为何失败）
- [ ] 未提交 `__pycache__` / `*.egg-info` / 大型 Feather / 大型 npz
- [ ] 新增 extras 依赖时同步更新 CI marker、README 与 `docs/from_source.md`
