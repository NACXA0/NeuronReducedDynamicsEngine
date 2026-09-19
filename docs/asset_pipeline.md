# A1 资产管线（D21）

Demo 卡（GIF / Colab / 离线包）是门面资产，必须像代码一样可复现、可 CI，否则随引擎演进静默腐烂。

## 三条脚本（均可独立运行）

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `scripts/render_demo.py` | 跑 preset → 写出 GIF/帧序列 + 版本水印 | 默认 headless Mock；真渲染需 MuJoCo offscreen + ffmpeg |
| `scripts/make_colab.py` | `examples/colab_src/demo.md` → `assets/demo/demo.ipynb` | 标准库生成 ipynb；可选 papermill 执行校验 |
| `scripts/make_release.py` | 聚合种子 artifacts + `manifest.json` → zip | 无重依赖 |

安装（要写出 GIF 时）：

```bash
uv sync --extra demo-assets
```

运行：

```bash
uv run python scripts/render_demo.py --preset flygym-demo-v01 --steps 40 --seed 0 --out assets/demo/
uv run python scripts/make_colab.py --out assets/demo/demo.ipynb
uv run python scripts/make_release.py --out dist/nrde-fly-offline.zip
```

## 硬约束

1. **`--seed` 驱动**：同 seed + 同版本 → 同画面；水印文件 `assets/demo/WATERMARK.txt` 含 `__version__`。
2. **headless 必选**（Q6 已决）：不依赖 X server，CI 可跑。
3. **CI**：`.github/workflows/ci-demo-assets.yml` — weekly + `workflow_dispatch`；**不进主 lane**。

## 腐烂防护（防漂移五件套之一）

每周 CI 用 papermill（若可用）执行 notebook；失败 = Demo 卡告警。无 papermill 时至少校验 ipynb JSON 可解析 + `nrde demo` smoke。

## 产物落点

```
assets/demo/          # 小 GIF / ipynb 可入库；大视频走 Release
dist/*.zip            # make_release 输出（gitignore）
```
