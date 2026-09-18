# 数据源清单（需求 §2.4 / RFC-001 D12）

| 管线 | 数据源 | 服务模块 | 对拟合 F 的作用 |
|---|---|---|---|
| 连接组侧 | MaleCNS Feather、FlyWire | DD-1 图结构、FR-1.5 权重、DD-6 α 标定 | 无（不产生 f-I） |
| 电生理侧 | ModelDB、NeuroElectro、文献单类记录 | DD-2 参数、DD-3 拟合原料 | 直接决定 F 质量 |

更多连接组 = 更多可标定回路，**不**增加 f-I 曲线数量。

## 本地路径

- 约定见 [`data/README.md`](../data/README.md)
- 下载脚本：`python scripts/download_malecns.py`（官方 GCS：`flyem-male-cns/v1.0/...`，CC-BY）
- 三件套：`connectome-weights-*.feather`、`body-annotations-*.feather`、`body-neurotransmitters-*.feather`
