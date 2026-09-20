# 自定义发放率管道（深度开发者）

引擎合同默认只有类型表。`place`（脑区/坐标）和 `modulator`（调质）已注册，默认恒等：没有数据时不改发放率，也不要求重训已发布 npz。要把它们变成非零修正，或追加新因素，用本页的管道，不要再交一张发放率来相加。关着的维必须与发布主线逐点相同。这是工具少维、探索加维（[RFC-003](../rfcs/RFC-003.md)），不是把生物 realism 做成默认字段。

图传播不变。换掉类型表之后，引擎仍是 \(I = W(r\Delta t)+I_{\mathrm{ext}}\)，\(r\) 由管道给出。那一步可以不再对应 ODE：同一核可以绑游戏/群落的类型映射，也可以用来量某根轴对 \(\Phi=F\circ W\) 的贡献。这不是发布用的生物拟合，也不构成「已形成智慧」。

当前管道是发放率后处理钩子，不是给 \(F\) 增加自变量。`r *= g(\mathrm{region},\mathrm{xyz})` 与 \(F(I,\theta)\) 不是一回事；真轴与扰动 API 见 [路线图](../roadmap.md) P2。

默认注册顺序：

```
I → type → place → modulator → 用户步 → r
```

`place` 与 `modulator` 恒等时，与只跑 `type` 逐点相同。深度 API 从子模块导入，不进顶层 `nrde`：

```python
from nrde.engine.ops import append_op, register_op, replace_op, reset_pipeline
```

## 1. 换教师 ODE（已有）

注册 `rhs` / `y0`，再 `offline fit`。见 [ModelSpec](../model_spec_protocol.md) 和 `examples/04_custom_model_pipeline.py`。这一步换的是 \(F_{\mathrm{type}}\) 的教师，不是管道顺序。

## 2. 加一个因素

先写恒等步（输入原样出去），注册后追加。有教师、有数据再改成非恒等。

```python
from nrde.engine.ops import append_op, register_op, reset_pipeline

class GainOp:
    name = "gain"
    g = 1.0

    def is_identity(self, fit):
        return abs(self.g - 1.0) <= 1e-15

    def apply(self, r, I, fit, region, xyz):
        if self.is_identity(fit):
            return r, 0
        return r * self.g, 0

reset_pipeline()
register_op(GainOp())
append_op("gain")
```

`g=1` 时 `step_rate` 与默认管道逐点相同。不要把未注册的函数写进 `datasets/fly/`，也不要把没有源 ODE 的步标成 `quality=ok`。

## 3. 固定系数改成函数

`PlaceMod` 的标量增益是常值函数。新因素不要往 `PlaceMod` 加字段，写成自己的 `RateOp`。同一步里 `g` 可以是数组，按节点、脑区或坐标变化：

```python
r * g(region, xyz)
```

向量化，不要对节点写 Python 循环。

## 4. 换掉现有某一步

```python
replace_op("place", "my_place")
replace_op("type", "linear_map")
```

`replace_op("type", ...)` 可以忽略拟合表，例如 `r = clip(k I, 0, rmax)`。图还在，F 不再是降阶 ODE。不要宣称 NFR-4 / NFR-5。跑完调用 `reset_pipeline()`，免得污染其它脚本。

禁止各步各出一张发放率再相加。未知步名会报错，并列出已注册的名字。

## 5. 口径

| | 发布主线（工具合同） | 自定义管道（探索维） |
|---|---|---|
| 用途 | 少维步进；果蝇是实例 | 开关维、抽象规则、非 ODE 的 \(F\) |
| 有效步骤 | `type`（其余恒等） | `append_op` / `replace_op` |
| 教师 | 内置 ODE / 以后的 ModelDB | 有则验收，无则不要标 ok |
| CUDA | 仅 `type` 之后全恒等、且无用户步 | 本轮走 NumPy |
| 禁止 | 把恒等维写进发布 npz | 把乘性滤镜说成 \(F\) 的新自变量；宣称智慧或新 AI |

完整示例：`uv run --no-sync python examples/05_custom_rate_op.py`。产物写到 `/tmp/nrde_fit/`，不碰种子 npz。
