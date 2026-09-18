# Python API (v0.1 / RFC-001 + RFC-002)

## 公共面（顶层 `import nrde`）

仅导出：`make` · `demo` · `run` · `offline` · `ModelSpec` · `__version__`。

```python
import nrde
env = nrde.make("flygym-demo-v01")
nrde.demo("flygym", steps=100)
nrde.offline(["fit", "--model", "adexp", "--type-id", "aCC"])
nrde.run(["simulate", "--fit", "artifacts/aCC.npz", "--steps", "50"])
```

深度 API 从子模块导入（不进顶层命名空间）。

## Offline (DD-3)

- `nrde.fitting.fit.fit_fi` / `fit_spike_lut` / `attach_lut` / `save_fit` / `load_fit` — L0/L1, NPZ schema v2 + `config_hash`
- `nrde.fitting.artifacts.resolve_artifact_path` / `migrate_directory` — 类型键路径 + 遗留模型键回退
- `nrde.fitting.chirp.chirp_impedance` / `decide_layer` — |Z(ω)| and L0–L2 assignment
- `nrde.fitting.srm.fit_srm_kernels` / `step_srm` — L2 exponential-sum IIR
- `nrde.fitting.pysr_backend.admission_exam` / `refine_fi_pysr` — optional; raises if Julia missing
- `nrde.specs.ModelSpec` — C-layer upload protocol

## Online (DD-4)

- `nrde.engine.rate.run_rate` / `step_rate`
- `nrde.engine.spike.run_spike` / `step_spike_lut` / `step_spike_izhikevich`
- `nrde.engine.hybrid.step_rate_mixed` — local Euler ODE fallback

## Data and calibration

- `nrde.io.connectome.load_connectome` / `erdos_renyi_graph` — Feather first
- `nrde.calibration.apply_shiu_weights` / `calibrate_alpha` — `W_syn = 0.275 mV`
- `nrde.validation.validate_single` / `validate_circuit` / `validate_hopkins_flywire`

## Embodiment (DD-5)

- `nrde.presets.make` / `demo` — A-layer
- `nrde.adapters.embodied.EmbodiedEnv` / `OpenLoopStimEnv`
- `nrde.adapters.flygym.NRDEFlyGymEnv`

CLI：`nrde demo` / `nrde run simulate` / `nrde offline fit|validate`（legacy：`fit` / `simulate` / `validate`）。

> 实现注记：内部拟合包目录为 `nrde.fitting`（避免与顶层符号 `nrde.offline` 同名冲突）；RFC-002 文中的 `offline/` 目录即此处。
