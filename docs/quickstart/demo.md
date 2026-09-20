# 跑果蝇 Demo（A1 / A2）

五分钟命令在仓库 [README](../../README.md)。本页是 Demo、Colab 和 GIF 的细节。果蝇是实例预设，不是引擎核（核是图 + 类型 \(F\) + 步进，见 [RFC-003](../rfcs/RFC-003.md)）。

> **不想装环境？** → [在 Colab 打开 Demo notebook](https://colab.research.google.com/github/NACXA0/NeuronReducedDynamicsEngine/blob/main/assets/demo/demo.ipynb)，或看 [`assets/demo/demo_seed0.gif`](../../assets/demo/demo_seed0.gif)。

## 主路径：A2 果蝇实例预设（Python）

给要在果蝇热点上快速开工的研究者（RL、回路建模）——开箱基线，不是奇观。

安装：

```bash
uv sync --extra fly
```

运行：

```bash
uv run nrde fetch fly --tier seed
uv run nrde demo flygym --steps 100 --headless
uv run python -c 'import nrde; print(nrde.demo("flygym", steps=100))'
```

内部：`artifacts/` 类型键种子 + `configs/flygym_demo.yaml` → `EmbodiedEnv` 闭环。  
口径：接口 Demo，**不**主张有意义行为（RFC-001 D5）。`headless` 为默认且为 A1 渲染管线依赖（D21 / Q6 已决）。

## A1 分支：果蝇奇观（非 Python）

| 产物 | 获取 |
|---|---|
| GIF | [`assets/demo/demo_seed0.gif`](../../assets/demo/demo_seed0.gif) |
| Colab | 上面的 notebook 链接 |
| 离线 zip | 发布之后的 GitHub Release 附件。仓库里没有这份 zip |

这些文件怎么重新生成，不在用户文档里。

## 离线包分支

Release 附件含种子 npz、`datasets/fly/` 和 `manifest.json`（sha256）。解压后：

安装：

```bash
uv sync --extra fly
```

运行：

```bash
uv run nrde demo flygym --config /path/to/extracted/flygym_demo.yaml --headless
```
