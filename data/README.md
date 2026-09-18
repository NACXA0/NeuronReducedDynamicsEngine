# 数据平面路径约定

本目录**不入库**原始连接组（体积大、许可证单独管理）。脚本会下载到下列路径：

```
data/
├── malecns/          # MaleCNS v1.0 Feather 三件套（FR-1.1 / RFC-001 D3）
│   ├── connectome-weights-male-cns-v1.0-minconf-0.5.feather
│   ├── body-annotations-male-cns-v1.0-minconf-0.5.feather
│   └── body-neurotransmitters-male-cns-v1.0.feather
└── flywire/          # 可选：FlyWire 导出（对照用，禁止与 MaleCNS 混用 ID）
```

下载：

```bash
python scripts/download_malecns.py
# 仅小文件（注释 + 递质，约 55 MB）：
python scripts/download_malecns.py --skip-weights
```

来源：https://male-cns.janelia.org/download/（CC-BY）。拟合用的电生理曲线**不**来自连接组，见 `docs/data_sources.md`。
