# 跑果蝇 Demo（A1 / A2）

> **不想装环境？** → A1：[Open in Colab](https://colab.research.google.com/)（notebook 由 `scripts/make_colab.py` 生成，M4 挂徽章）· 或看 `assets/demo/` GIF。

## 主路径：A2 果蝇实例预设（Python）

给要在果蝇热点上快速开工的研究者（RL、回路建模）——开箱基线，不是奇观。

```bash
pip install "neuron-reduced-dynamics-engine[fly]"
nrde fetch fly --tier seed          # 下载/校验种子 artifacts（manifest）
nrde demo flygym --steps 100 --headless
# 或
python -c 'import nrde; print(nrde.demo("flygym", steps=100))'
```

内部：`artifacts/` 类型键种子 + `configs/flygym_demo.yaml` → `EmbodiedEnv` 闭环。  
口径：接口 Demo，**不**主张有意义行为（RFC-001 D5）。`headless` 为默认且为 A1 渲染管线依赖（D21 / Q6 已决）。

## A1 分支：果蝇奇观（非 Python）

| 产物 | 获取 | 构建 |
|---|---|---|
| GIF / 短视频 | README Demo 卡 · `assets/demo/`（小 GIF 可入库） | `scripts/render_demo.py` |
| Colab | README 徽章 | `scripts/make_colab.py` ← `examples/colab_src/demo.md` |
| 离线 zip | GitHub Release | `scripts/make_release.py` |

详见 [asset_pipeline.md](../asset_pipeline.md)。

## 离线包分支

Release 附件含种子 npz + `manifest.json`（sha256）。解压后：

```bash
pip install "neuron-reduced-dynamics-engine[fly]"
nrde demo flygym --config /path/to/extracted/flygym_demo.yaml --headless
```
