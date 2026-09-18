# FRE — 果蝇降维动力学引擎
输入连接组，输出**绑定神经元类型的降维函数F**（L0‑L2层级）。离线阶段可不受算力限制（支持样条函数、SRM核函数、可选PySR）；在线推理开销极低：稀疏收集‑散射运算 + 查表 / 一阶无限冲激响应（IIR）。

v0.1版本达成目标：*拟合同AdExp类函数F，输出相对于常微分方程（ODE）的可量化误差；基于Shiu论文的突触权重`W_syn = 0.275 mV`完成α参数标定；运行子神经图；在具身仿真环境`EmbodiedEnv`执行100个有限时间步。* 本版本**并不代表**完整中枢神经系统生物物理重建，也不实现有实际意义的运动行走仿真。

## 关于 PySR？
PySR仅用作**可选的后处理优化器**，不作为默认训练器（项目仓库：[MilesCranmer/PySR]([https://github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR))，参考RFC‑001 D9）：
- 适用安全区间：维度 ≤ 2，采样点 ≤ 10⁴，`maxsize ≤ 20`，**仅限离线使用**。
- L0层默认仍然采用PCHIP分段三次埃尔米特样条（基电流阈值不具备全局解析解；傅里叶基会出现吉布斯振铃效应）。
- L2层所说的“多函数”指**SRM核函数** κ、η、θ，而非一堆互不关联的多层感知机（MLP）。
- 执行`pip install fre‑neuron‑engine`时**禁止自动安装Julia**（需求NFR‑11）。如需该扩展：`pip install 'fre‑neuron‑engine[pysr]'`。
- 准入校验：64点LIF神经元f‑I（频率‑电流）闭式解测试。测试失败 → 自动停用PySR模块（详见文档 [docs/pysr_admission.md](docs/pysr_admission.md)）。

## 项目定位
| 项目 | 实现能力 | FRE项目说明 |
|---|---|---|
| FlyBrainLab / Neurokernel | 在线 Hodgkin‑Huxley / LIF 仿真 | FRE不在网络主循环内运行HH模型 |
| ornata/fly | 连接组 + 通用激活函数 | FRE的函数F由指定常微分方程/SRM拟合得到 |
| Shiu 等人 2024 | FlyWire统一LIF模型，固定`W_syn`突触权重 | 该论文的`W_syn`作为我们α参数**初始值**；额外增加类型绑定的F函数 |
| `flybrain` GPU LIF | 雄性果蝇中枢神经系统166k神经元，单步约1.4 ms | 仅做压力负载测试，非最终产品 |

对外表述口径：**不同神经元类型可绑定各自不同的F函数**。禁止描述为“已重建多个具备生物物理精度的神经元细胞类型”。

连接组数据集**本身无法生成f‑I频率‑电流曲线**；该类数据来源于电生理实验（ModelDB数据库、相关文献）。详见 [docs/data_sources.md](docs/data_sources.md)。

## 安装方式
```bash
pip install -e ".[dev]"          # 不包含Julia依赖
# pip install -e ".[pysr]"       # 可选依赖；CI流水线允许该模块测试失败
```

## 快速上手
```bash
fre fit --model adexp --out artifacts/adexp.npz
fre validate --pysr-exam
python examples/01_single_neuron_fit.py
python examples/02_small_circuit.py
python examples/03_flygym_closed_loop.py
```

## 层级架构（DD‑3）
| 层级 | 输出产物 | 运行时开销 |
|---|---|---|
| L0 | 一维响应函数 r(I) | O(1) 插值运算 |
| L1 | 二维查找表 LUT (I, Δt) | O(1) 查表 |
| L2 | SRM核函数 κ, η, θ（IIR无限冲激响应） | 每个核O(1) |
| L3 | FNO / Volterra模型 | v0.2版本才会实现 |

频域响应`|Z(ω)|`存在峰值的神经元类型会升级至L2层级。凡是判定无法适配的场景，降级使用在线欧拉积分（`step_rate_mixed`，详见 [docs/known‑inapproximable.md](docs/known‑inapproximable.md)）。

相关参考工作：Shiu 等人 2024；Jolivet/Gerstner 的SRM模型；Brette & Gerstner 2005；Kobayashi 2009 MAT；Cranmer PySR；FlyBrainLab。

```bash
pytest --cov=fre --cov-fail-under=70 -m "not pysr"
```

> 术语注解：
> - SRM：尖峰响应模型 Spike Response Model
> - IIR：无限冲激响应滤波器
> - LIF：泄漏积分发放神经元
> - AdExp：指数自适应神经元
> - PCHIP：保形分段三次埃尔米特插值
> - Connectome：神经连接组
> - f‑I curve：频率‑电流曲线，描述神经元输入电流与发放频率关系

