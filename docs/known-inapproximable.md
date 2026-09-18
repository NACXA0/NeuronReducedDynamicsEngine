# 降阶不适用清单（DD-4 / RFC-001 D10）

一等公民文档。验证失败的回路记在这里，不在引擎里静默掩盖。仿真可通过 `fallback_types` 对这些类型在线 Euler ODE。

| ID | 回路 / 模型 | 失败模式 | 建议层级 |
|---|---|---|---|
| F-HH-LUT | 标准 HH + 2D `(I, Δt)` LUT | 门控变量被压掉；burst / 去极化阻滞对不齐 | 升 L2 SRM 或永久回退 ODE |
| F-TYPE-II | HH Type-II f-I | CubicSpline 在发放间隙写出不可能的中间频率 | L0 用 PCHIP；I < I_onset 强制 0 Hz |
| F-SYNC | 依赖同步或延迟的环（CX 环共振） | 静态 f-I 丢失相位 | L2；否则 `fallback_types` 在线 ODE |
| F-PYSR | PySR 准入考试 | 本环境未安装 Julia/PySR | 退场；L0 样条 / L2 指数核保留 |

蘑菇体输出与 SEZ 味觉为优先验证回路；尚未用真实连接组跑通的条目保持开放。
