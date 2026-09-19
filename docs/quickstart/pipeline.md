# 拟合 F（L0 / L1 / L2）：怎么开始

实现包是 **`nrde.fitting`**。命令是 **`uv run nrde offline fit`**。

这一步只拟合降阶函数 F，不读连接组，也不做稀疏矩阵乘。乘 `W r` 是下一步，见 [稀疏传播](engine.md)。v0.1 不做网络级训练，也不把 TensorBoard / SwanLab 接到这条命令上：一次拟合没有 epoch，只有退出时的一行 JSON 和一份 npz。

F 的轴是细胞状态，不是图。少留几轴就是更低的层。L3（FNO / Volterra）属 v0.2，下面没有。

| 示例 | 层 | 开关 | 写出 |
|---|---|---|---|
| 只留稳态发放率 | L0 | 无 | `lif_l0.npz` |
| 再加上距上次脉冲的时间 | L1 | `--lut` | `lif_l1.npz` |
| 再加 SRM 核 | L2 | `--srm` | `lif_l2.npz` |
| LUT 和 SRM 都要 | L1 的表 + L2 的核 | `--lut --srm` | `lif_l1l2.npz` |

`--chirp` 可以加在任一条上。它扫阻抗，有峰就在 JSON 的 `layer` 里建议升到 L2。它不是第四层。

PySR 只精修已经拟合好的 f–I，不是启动前提。没装 Julia 也可以拟合。

## 1. 安装

不要装进系统 Python。在仓库根目录：

```bash
uv sync --extra dev
```

Numba 是可选的同一套标量核，装了只加速，不改变拟合结果。只有要跑 PySR 准入考试时才 `uv sync --extra pysr`，并且本机要有 Julia。

## 2. 四份示例

产物不要写进仓库里的种子文件。`artifacts/aCC.npz`、`type_0.npz`、`hh_demo.npz` 的 sha256 写在 `configs/manifests/fly.yaml` 里，覆写之后 `uv run nrde fetch fly` 会失败。

下面四条都是 LIF、电流 0 到 0.8 nA、16 个采样点。标准输出一行 JSON：`out`、`r2`、`quality`、`layer`、`config_hash`。`quality` 为 `ok` 时退出码是 0。`r2 ≥ 0.98` 才算过 NFR-4。

### L0

```bash
mkdir -p /tmp/nrde_fit
uv run nrde offline fit \
  --model lif \
  --type-id lif_l0 \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --out /tmp/nrde_fit/lif_l0.npz
```

### L1

```bash
uv run nrde offline fit \
  --model lif \
  --type-id lif_l1 \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --lut \
  --out /tmp/nrde_fit/lif_l1.npz
```

### L2

```bash
uv run nrde offline fit \
  --model lif \
  --type-id lif_l2 \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --srm \
  --out /tmp/nrde_fit/lif_l2.npz
```

### L1+L2

```bash
uv run nrde offline fit \
  --model lif \
  --type-id lif_l1l2 \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --lut \
  --srm \
  --chirp \
  --out /tmp/nrde_fit/lif_l1l2.npz
```

换模型时改 `--model` 和 `--type-id`，仍然写到 `/tmp/nrde_fit/`。`toy_leak` 要先注册，见 `examples/04_custom_model_pipeline.py`。省略 `--out` 时写到 `artifacts/{type_id}.npz`；不要用已入库的 `type_id`（`aCC`、`type_0`、`hh_demo`）。

## 3. 检查单胞精度

L0 只报告 f–I 的 R²。带了 `--lut` 的 L1、L1+L2 还会报 1 秒窗口上的 Victor–Purpura 相对距离（`q = 0.1 / ms`，≤ 10% 为通过）。

```bash
uv run nrde offline validate --fit /tmp/nrde_fit/lif_l1.npz
```

把路径换成另外三份即可。比较用的是 L1 时刻表，不是 1 ms 步进的光栅。

## 4. 在 Python 里做同一件事

```python
from nrde.fitting.fit import attach_lut, fit_fi, fit_spike_lut, save_fit
from nrde.fitting.srm import attach_srm, fit_srm_kernels

fit = fit_fi("lif", type_id="lif_l0", I_min=0.0, I_max=0.8, n_I=16)
save_fit(fit, "/tmp/nrde_fit/lif_l0.npz")

fit = attach_lut(fit, fit_spike_lut("lif", I_min=0.0, I_max=0.8, n_I=16))
save_fit(fit, "/tmp/nrde_fit/lif_l1.npz")

fit = attach_srm(fit, fit_srm_kernels("lif"))
save_fit(fit, "/tmp/nrde_fit/lif_l1l2.npz")
```

用 `uv run python` 执行。自定义 ODE 先 `ModelSpec.register()`，再把名字传给 `fit_fi`。协议见 [ModelSpec](../model_spec_protocol.md)。

重复拟合且 `config_hash` 相同则不覆写文件。要强制重写加 `--force`。

## 5. 参数对照

| 参数 | 含义 | v0.1 |
|---|---|---|
| `--model` | 已注册的 ODE 名 | 已支持。内置 `lif` / `explif` / `adexp` / `hh` / `izhikevich` |
| `--type-id` | 类型键，也是默认文件名 | 已支持 |
| `--out` | npz 路径 | 已支持。请显式写出，避开种子 |
| `--I-min` / `--I-max` / `--n-I` | f–I 采样 | 已支持。`--current-range auto` 还没有 |
| `--lut` | L1：二维 LUT + 恒定电流时刻表 | 已支持 |
| `--srm` | L2：SRM 核 | 已支持 |
| `--chirp` | 扫频阻抗，参与判层 | 已支持。可加在上面任一条 |
| `--layers L0,L1,L2` | 一次声明各层 | **还没有**。用 `--lut` / `--srm` |

不需要 FlyGym，也不需要果蝇连接组。下一步：[稀疏传播](engine.md)。
