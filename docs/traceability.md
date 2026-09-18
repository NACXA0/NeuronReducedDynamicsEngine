# FR ↔ DD ↔ TC

| 需求 | 设计 | 测试 |
|---|---|---|
| FR-1.1–1.5 | DD-1 `nrde.io.connectome`（Feather 优先，Shiu α） | TC-1 `tests/test_connectome.py` |
| FR-2.1–2.3 | DD-2 `nrde.models`（AdExp 为 v0.1 主对象） | TC-2 `tests/test_models.py` |
| FR-3.1–3.9 | DD-3 L0–L2：`nrde.fitting`（fit / chirp / srm / pysr） | TC-3 `tests/test_fit.py`、`tests/test_rfc001.py` |
| FR-4.1–4.6 | DD-4 `nrde.engine` + `hybrid` 局部 ODE 回退 | TC-4 `tests/test_engine.py` |
| FR-5.1–5.3 | DD-5 `EmbodiedEnv` / FlyGym / OpenLoop | TC-5 `tests/test_flygym.py` |
| FR-6.1–6.3 | DD-6 Shiu α 标定；FlyWire-only Hopkins | TC-6 `tests/test_validation.py`、`tests/test_calibration.py` |
| NFR-1–3 | DD-4 | `tests/test_perf.py`（报告型；GPU 倍数不在 CI 强制） |
| NFR-4–5 | DD-3 / DD-6 | TC-3 / TC-6；VP `q = 1/10 ms`，窗口 1 s |
| NFR-8 | DD-5 | 核心 import 不得加载 flygym |
| NFR-11 | PySR 可选 extra | `docs/pysr_admission.md`；CI `pysr` lane |
| NFR-12 | L2 IIR O(1)/核 | `tests/test_rfc001.py` |

## RFC-002 决策追溯（D13–D23）

| 编号 | 落点 | 测试 / 文档 |
|---|---|---|
| D13 | CI 3.11 + 3.14 allow-failure | `.github/workflows/ci.yml` |
| D14 | `offline/` → `fitting/` | import 路径；CLI 仍 `offline` |
| D20 | `nrde[fly]` + fetch manifest | `nrde fetch`；`configs/manifests/fly.yaml` |
| D21 | A1 资产管线 | `docs/asset_pipeline.md`；`ci-demo-assets.yml` |
| D22 | 命名映射框 | README |
| D23 | 历史文档横幅 + CHANGELOG | 中文四文档首行；`docs/CHANGELOG.md` |

标记语法（自动化预留）：测试名或文档锚点使用 `D##` / `FR-x` / `TC-x`；脚本扫描尚未强制。

Hopkins / Shiu 对照只用 FlyWire 子集。MaleCNS 全图只做加载与速率一步。连接组数据不产生 f-I（见 `data_sources.md`）。
