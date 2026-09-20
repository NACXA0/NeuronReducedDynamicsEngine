# 文档地图

仓库首页 [README](https://github.com/NACXA0/NeuronReducedDynamicsEngine) 是展示面：介绍、实测数字、五分钟命令、本目录的索引。细节不进 README。

对外口径：**types may bind different F**。引擎默认只有类型这一身份维，核是图 + \(F\) + 步进。不说已重建 N 种真实动力学，也不主张有意义行走，更不说已形成智慧或已得到新 AI。宣传数字的算法在 [证据链](reports/evidence.md)。终局与改进顺序在 [RFC-003](rfcs/RFC-003.md) / [路线图](roadmap.md)。

## 怎么读

1. 先跑通 README 的安装 / 运行 / 测试。
2. 要自己拟合或换层，看 [拟合 F](quickstart/pipeline.md) 再看 [稀疏传播](quickstart/engine.md)。跑通 aCC 烟测后，同一条命令换类型名在该页 §4。
3. 要把核当基础设施，或加探索维、换掉 \(F\)，看 [自定义管道](quickstart/extend.md)。已发布表不必因脑区/坐标占位而重训。
4. 冲突：禁止项以 [RFC-001](rfcs/RFC-001.md) 为准；产品面以 [RFC-002](rfcs/RFC-002.md) 为准；终局与工具/探索切分以 [RFC-003](rfcs/RFC-003.md) 为准。中文规格是历史快照。发布、PyPI、数据地址和静态宣传页不在本目录，见仓库根目录 `maintainer/`。宣传页源码在 `maintainer/WEB`（React 19，附 Caddyfile 示例），不是文档站。

## 结构（当前）

| 层 | 主线（工具合同） | 不在主线（探索开关 / 延期） |
|---|---|---|
| 身份 | 类型 → 一张 \(F\) | 脑区、坐标、调质（无数据；管道步默认恒等） |
| 状态 | L0 电流；L1 时刻表；L2 SRM | L3 / FNO（`deferred_v0.2`） |
| 图 | \(W\)、递质、\(I_{\mathrm{ext}}\) | STP、延迟 |
| 教师 | 内置 ODE | ModelDB / FBL 槽位，尚未换表 |
| 用法 | 只读步进、果蝇实例 | 可学习 \(\Phi\)（v0.1 不做；见路线图 P3） |

## 页面

快速开始：[Demo](quickstart/demo.md) · [拟合](quickstart/pipeline.md) · [传播](quickstart/engine.md) · [从源码构建](from_source.md)

深度开发：[自定义管道](quickstart/extend.md) · [ModelSpec](model_spec_protocol.md) · [API](api.md) · [意图与改进计划](roadmap.md)

数据：[数据源](data_sources.md) · [证据链](reports/evidence.md) · [M1 索引](reports/README.md)

规格：[RFC-001](rfcs/RFC-001.md) · [RFC-002](rfcs/RFC-002.md) · [RFC-003](rfcs/RFC-003.md) · [需求](需求文档.md) · [计划书](项目计划书.md) · [设计说明书](系统设计说明书.md) · [初始设想](初始设想.md)
