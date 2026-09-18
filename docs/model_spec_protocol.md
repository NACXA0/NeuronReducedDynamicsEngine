# ModelSpec 协议（C 层入口）

**协议版本**：`PROTOCOL_VERSION = 1`（`nrde.specs`）  
**对应**：RFC-002 §3.3 协议 1；破坏性变更走 RFC（R11）

## 最小接口

```python
from dataclasses import dataclass
import numpy as np
from nrde import ModelSpec

@dataclass
class MyParams:
    tau: float = 10.0
    V_rest: float = -70.0
    Vth: float = -50.0
    Vreset: float = -70.0

def rhs(t, y, I, p): ...
def y0(p): ...

spec = ModelSpec(name="mine", state_dims=1, rhs=rhs, y0=y0, Params=MyParams)
spec.register()
```

强制：`rhs` / `y0`。可选：`reset`、`spike_threshold`、`fitted_targets`（`f-i` / `kernels` / `H_omega`）。

内置 LIF / ExpLIF / AdExp / HH / Izhikevich 均为参考实现：`nrde.specs.model_spec_from_registry("adexp")`。

## 管线入口

```bash
nrde offline fit --model mine --type-id mine --out artifacts/mine.npz
# 或
python examples/04_custom_model_pipeline.py
```

产物 meta 含 `config_hash`：重复拟合且哈希相同则跳过写入（幂等）。

## IO 点声明（协议 3）

见 `configs/flygym_demo.yaml` 的 `io_map`：按 `type_id` 选择编码器/解码器；研究者可改用 `neuron_id` 列表（后续扩展）。
