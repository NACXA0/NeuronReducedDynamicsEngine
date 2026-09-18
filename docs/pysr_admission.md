# PySR 准入考试记录（RFC-001 D9.5 / NFR-11）

| 日期 | 环境 | 结果 | 说明 |
|---|---|---|---|
| 2026-09-18 | CPython 3.14, no Julia | **retired** | `pysr` 未安装。主包按 NFR-11 不引入 Julia。`fre validate --pysr-exam` 记录退场，拟合走样条 / SRM 核。 |

恢复条件：`pip install 'fre-neuron-engine[pysr]'` 后，用 LIF 解析 f-I（64 点）在 30 分钟预算内得到含 `log` 或有理式结构的表达式，R² ≥ 0.95。不通过则整条 PySR 路径保持退场。

PySR 只做 L0/L1/L2 数值产物的离线精化器（maxsize ≤ 20，维数 ≤ 2，样本 ≤ 10⁴），不做在线训练。
