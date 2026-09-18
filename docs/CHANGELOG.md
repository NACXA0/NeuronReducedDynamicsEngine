# Changelog

## 0.1.0 — 工程结构补齐

- 开源文件：`CONTRIBUTING.md`、`CITATION.cff`；README 写清仓库 / PyPI / import 三名映射。
- 数据平面：`data/` 路径约定 + `scripts/download_malecns.py` / `make_figures.py`。
- 验证报告落点：`docs/reports/`；mkdocs nav 覆盖中文规格与 reports。
- CI 三 lane：`ci.yml`（核心）/ `ci-pysr.yml` / `ci-flygym.yml`；`tests/conftest.py` 共享 fixture。
- 产物 meta 增加 `type_ids`、`git_commit`、`config_hash`；示例更名为 `03_flygym_interface_demo.py`。
- 可复现：`uv.lock`；开发可跟进 Python 3.14，CI 主 lane 仍为 3.11。

## 0.1.0 — RFC-001

- L0–L2 fitting: AdExp is the M1 primary object; chirp |Z(ω)|; SRM IIR; PySR optional extra (retired without Julia).
- α starts at Shiu W_syn = 0.275 mV (`SHIU_ALPHA_NA`).
- `EmbodiedEnv` + FlyGym / OpenLoop backends; 100-step interface demo.
- Local ODE fallback + `docs/known-inapproximable.md`.
- NPZ schema v2 (backward compatible with v1).


## 0.1.0 — M0–M5 scaffold

- M0: Feather as MaleCNS primary format; FR↔DD↔TC table; FlyGym v2 hooks; VP `q = 1/10 ms`.
- M1: ExpLIF + AdExp/HH offline f-I with PCHIP gate (NFR-4).
- M2: Sparse rate engine, α calibration, uniform-LIF ablation.
- M3: 2D spike LUT, circuit vs Euler ODE, Izhikevich 2D fallback (R3).
- M4: `FREFlyGymEnv` 100-step closed loop (mock or FlyGym v2).
- M5: MaleCNS-like Feather load-and-step, MkDocs, coverage ≥70%.
