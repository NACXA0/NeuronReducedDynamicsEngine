# 拟合你自己的模型（C 层）

实现包：**`nrde.fitting`**。CLI：`nrde offline …`。

```bash
pip install neuron-reduced-dynamics-engine
python examples/04_custom_model_pipeline.py
# 注册 ModelSpec 后：
nrde offline fit --model toy_leak --type-id toy_leak
```

## 协议 2：管线参数（骨架）

| 参数 | 含义 | v0.1 |
|---|---|---|
| `--model` / ModelSpec 名 | 已注册 ODE | 已支持 |
| `--type-id` | 类型键产物名 | 已支持（默认 `artifacts/{type_id}.npz`） |
| `--layers L0,L1,L2` | 产出层级；L2 触发 chirp 判据 | **骨架**：现用 `--lut` / `--srm` / `--chirp`；统一 `--layers` 跟进 |
| `--current-range auto` | rheobase 二分定采样区间 | **计划中**（暂用 `--I-min` / `--I-max`） |

协议说明：[ModelSpec](../model_spec_protocol.md)。  
不需要 FlyGym、不需要果蝇连接组；`type_model_map.yaml` 仅作示例。
