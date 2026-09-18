# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号语义化。决策编号见 [RFC-002](rfcs/RFC-002.md)。

## [Unreleased]

### Added
- A1/A2 产品面拆分；`nrde[fly]` extras；`nrde fetch` + `configs/manifests/fly.yaml`
- A1 资产管线骨架：`scripts/render_demo.py` / `make_colab.py` / `make_release.py`；`docs/asset_pipeline.md`；`ci-demo-assets.yml`
- CI：Python 3.14 allow-failure lane（D13）
- 从源码构建清单：`docs/from_source.md`（extras / 系统依赖 / 数据集）；README 短链

### Changed
- README / docs 安装矩阵改为 A1/A2/B/C 四行；历史中文规格加 RFC 横幅（D23）
- 品牌锁定为 NRDE / `nrde` / `neuron-reduced-dynamics-engine` / `NeuronReducedDynamicsEngine`（果蝇仅为实例预设；方法术语为 state approximation）
- 品牌守卫：`scripts/assert_nrde_brand.py` + `tests/test_brand_naming.py`

## [0.1.0] — 2026-09-18

### Added
- RFC-001 / RFC-002 基线：L0–L2、EmbodiedEnv、类型键 artifacts、公共五符号 API
- D13–D23（节选）：`fitting/` 改名、CI 三 lane、config_hash 幂等、migrate 脚本、命名映射

### Notes
- PyPI 名 `neuron-reduced-dynamics-engine`（Q9 发布前查重）；开发可用 Python 3.14，CI 主 lane 3.11
