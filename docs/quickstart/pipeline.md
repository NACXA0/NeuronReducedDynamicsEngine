# 拟合管线（C 层）：怎么开始

实现包是 **`nrde.fitting`**。命令行入口是 **`nrde offline`**。

这条管线是离线拟合，不是网络训练，也不是在线学习。v0.1 不做 STDP、不做回路级训练。它用你注册的 ODE 生成样本，再拟合降阶的 F：

| 层 | 得到什么 | 怎么开 |
|---|---|---|
| L0 | f–I 曲线 | `nrde offline fit` |
| L1 | 恒定电流下的发放时刻表，外加二维一步响应 LUT | 加上 `--lut` |
| L2 | SRM 核（κ、η、θ 的指数和） | 加上 `--srm` |
| 判层 | chirp 阻抗 \|Z(ω)\|，决定停在哪一层 | 加上 `--chirp` |

PySR 只精修已经拟合好的 f–I，不是启动这条管线的前提。没装 Julia 也可以拟合。

## 1. 装好再跑

核心安装不依赖 FlyGym、Julia 或连接组：

```bash
pip install -e ".[dev]"
# 已用 uv 时：uv sync --extra dev
```

Numba 是可选的同一套标量核，装了只加速，不改变拟合结果。只有要跑 PySR 准入考试时才需要 `.[pysr]` 和 Julia。

## 2. 第一条拟合命令

产物不要写进仓库里的种子文件。`artifacts/aCC.npz`、`type_0.npz`、`hh_demo.npz` 的 sha256 写在 `configs/manifests/fly.yaml` 里，覆写之后 `nrde fetch fly` 会失败。

```bash
mkdir -p /tmp/nrde_fit
nrde offline fit \
  --model lif \
  --type-id lif_demo \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --lut \
  --chirp \
  --out /tmp/nrde_fit/lif.npz
```

标准输出是一行 JSON：`out`、`r2`、`quality`、`layer`、`config_hash`。`quality` 为 `ok` 时进程退出码是 0。`r2 ≥ 0.98` 才算过 NFR-4。

同一命令换模型即可：

```bash
nrde offline fit --model adexp --type-id aCC_local --lut --srm --chirp \
  --out /tmp/nrde_fit/aCC_local.npz
nrde offline fit --model toy_leak --type-id toy_leak \
  --out /tmp/nrde_fit/toy_leak.npz
```

`toy_leak` 要先注册，见 `examples/04_custom_model_pipeline.py`。

省略 `--out` 时写到 `artifacts/{type_id}.npz`。拟合自己的类型可以用这个默认路径；不要用已入库的 `type_id`（`aCC`、`type_0`、`hh_demo`）。

## 3. 检查单胞精度

```bash
nrde offline validate --fit /tmp/nrde_fit/lif.npz
```

JSON 里的 `nfr4_pass` 是 f–I 的 R²。带了 `--lut` 时还有 `relative_vp` 和 `nfr5_pass`：1 秒窗口、Victor–Purpura 代价 `q = 0.1 / ms`，相对 `(n_ref + n_approx)` 的距离 ≤ 10% 才算通过。

比较用的是 L1 时刻表（沿电流轴插值发放间隔），不是 1 ms 步进的光栅。二维 LUT 仍给有突触电流的在线步进用。

## 4. 在 Python 里做同一件事

```python
from nrde.fitting.fit import attach_lut, fit_fi, fit_spike_lut, save_fit

fit = fit_fi("lif", type_id="lif_demo", I_min=0.0, I_max=0.8, n_I=16, chirp=True)
fit = attach_lut(fit, fit_spike_lut("lif", I_min=0.0, I_max=0.8, n_I=16))
save_fit(fit, "/tmp/nrde_fit/lif.npz")
```

L2：`fit_srm_kernels` + `attach_srm`（`nrde.fitting.srm`）。自定义 ODE 先 `ModelSpec.register()`，再把名字传给 `fit_fi`。协议见 [ModelSpec](../model_spec_protocol.md)。

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
| `--chirp` | 扫频阻抗，参与判层 | 已支持 |
| `--layers L0,L1,L2` | 一次声明各层 | **还没有**。用上面三个开关 |

不需要 FlyGym，也不需要果蝇连接组。`configs/type_model_map.yaml` 只是类型到模型的示例。
