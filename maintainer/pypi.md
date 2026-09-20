# 构建与发布 PyPI

发行名是 `neuron-reduced-dynamics-engine`（RFC-002 Q9）。import 仍是 `nrde`。wheel 只打包 `src/nrde`，不含 `datasets/`、`artifacts/`、`configs/`。装好之后找表，靠 `nrde fetch` 和清单里的下载地址，不靠 wheel。

先发 [GitHub Release](release.md) 并写好 `downloads.primary`，再上传 PyPI。顺序反了的话，PyPI 用户会得到一个找不到表的包。

## 上传前

```bash
uv sync --extra dev
uv run --extra dev pytest -m "not pysr and not flygym"
uv run --extra dev ruff check src tests examples scripts
```

核对版本号只改这两个地方，而且相同：

- `pyproject.toml` → `version`
- `src/nrde/__init__.py` → `__version__`

PyPI 上不能覆盖已有版本。传错了就升一位再传。

## 构建

```bash
uv build
```

产物在 `dist/`：`neuron_reduced_dynamics_engine-<version>-py3-none-any.whl` 和同名 `.tar.gz`。`dist/` 已 gitignore。这和离线数据 zip 不是同一个文件，不要把 `nrde-fly-offline.zip` 传上 PyPI。

检查 wheel 里没有 npz：

```bash
uv run python -c 'import zipfile; from pathlib import Path; whl=next(Path("dist").glob("*.whl")); print([n for n in zipfile.ZipFile(whl).namelist() if n.endswith(".npz")])'
```

输出应是 `[]`。

## 上传

需要 PyPI API token，环境变量 `UV_PUBLISH_TOKEN`。不要把 token 写进仓库。

先传 TestPyPI：

```bash
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
```

在空目录里用 TestPyPI 装一次（索引还要能解析 numpy 等依赖时，加上 `--extra-index-url https://pypi.org/simple`）。然后：

```bash
nrde fetch fly --tier pretrained
```

应打印已下载或已校验，且退出码为 0。这一步失败就不要传正式 PyPI。

正式上传：

```bash
uv publish dist/*
```

上传不会创建 GitHub Release，也不会更新 `downloads.secondary`。两件事分开做。
