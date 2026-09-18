# 贡献指南

感谢参与 FRE（Fly Reduced-dynamics Engine）开发。

## 命名约定

| 角色 | 名称 |
|---|---|
| Git 仓库 | `StateApproxNeur` |
| PyPI / 发行名 | `fre-neuron-engine` |
| Python import | `fre` |

三者可以不同，但请勿再引入第四个名字。本地 `*.egg-info` 由 setuptools 生成，**不要提交**。

## 环境

- 推荐 Python **3.11–3.14**（开发环境可跟进最新稳定版；CI 主 lane 使用 3.11 以保证可选依赖 wheel 可用）。
- 安装：`pip install -e ".[dev]"` 或 `uv sync --extra dev`
- 可选：`.[pysr]`、`.[flygym]`、`.[docs]`

## 开发流程

```bash
pre-commit install
ruff check src tests examples
pytest --cov=fre --cov-fail-under=70 -m "not pysr and not flygym"
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
- [ ] 新增 extras 依赖时同步更新 CI marker 与 README
