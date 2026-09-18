# 跑一次连接组仿真（B 层）

拟合实现位于包 **`nrde.fitting`**（CLI 动词仍叫 `offline`，RFC-002 D14）。

```bash
pip install neuron-reduced-dynamics-engine
nrde offline fit --model adexp --type-id aCC
nrde run simulate --fit artifacts/aCC.npz --n 100 --steps 50
```

自备 Feather：

```bash
nrde fetch fly --tier full          # 或 scripts/download_malecns.py
nrde run simulate --graph data/malecns/connectome-weights-....feather \
  --annotations data/malecns/body-annotations-....feather \
  --fit artifacts/aCC.npz --steps 100
```

类型绑定见 `configs/type_model_map.yaml`；α 起点见 `configs/alpha_defaults.yaml`（Shiu 0.275 mV）。
