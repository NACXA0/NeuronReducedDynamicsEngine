# 重新生成 Demo 资产

GIF、Colab notebook、离线 zip 必须能复现，否则引擎改了而门面还是旧的。用户只消费 `assets/demo/` 和 Release 附件。生成步骤在这里。

## 三条脚本

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `scripts/render_demo.py` | 跑 preset，写 GIF / 帧和版本水印 | 默认 headless Mock。真渲染要 MuJoCo offscreen 和 ffmpeg |
| `scripts/make_colab.py` | `examples/colab_src/demo.md` → `assets/demo/demo.ipynb` | 标准库写 ipynb |
| `scripts/make_release.py` | 种子 + `datasets/fly/` → zip | 无重依赖。挂到 Release 的步骤见 [release.md](release.md) |

```bash
uv sync --extra demo-assets
uv run python scripts/render_demo.py --preset flygym-demo-v01 --steps 40 --seed 0 --out assets/demo/
uv run python scripts/make_colab.py --out assets/demo/demo.ipynb
uv run python scripts/make_release.py --out dist/nrde-fly-offline.zip
```

约束：

1. `--seed` 相同且版本相同则画面相同。`assets/demo/WATERMARK.txt` 含 `__version__`。
2. headless 是默认。不依赖 X server，CI 能跑。
3. `.github/workflows/ci-demo-assets.yml` 每周、手动，以及 Release 发布时运行。它不进主 CI。失败只告警（`continue-on-error`）。
4. 小 GIF 和 ipynb 可以提交。大视频不要提交，走 Release。`dist/*.zip` 不要提交。

改 Colab 文案时改 `examples/colab_src/demo.md`，再跑 `make_colab.py`。不要只改生成的 ipynb。

公开的 Colab 链接指向仓库里的 notebook：

```text
https://colab.research.google.com/github/NACXA0/NeuronReducedDynamicsEngine/blob/main/assets/demo/demo.ipynb
```

换默认分支时同时改 `docs/quickstart/demo.md` 里的这个链接。
