# Python API (v0.1 / RFC-001)

## Offline (DD-3)

- `fre.offline.fit.fit_fi` / `fit_spike_lut` / `attach_lut` / `save_fit` / `load_fit` — L0/L1, NPZ schema v2
- `fre.offline.chirp.chirp_impedance` / `decide_layer` — |Z(ω)| and L0–L2 assignment
- `fre.offline.srm.fit_srm_kernels` / `step_srm` — L2 exponential-sum IIR
- `fre.offline.pysr_backend.admission_exam` / `refine_fi_pysr` — optional; raises if Julia missing

## Online (DD-4)

- `fre.engine.rate.run_rate` / `step_rate`
- `fre.engine.spike.run_spike` / `step_spike_lut` / `step_spike_izhikevich`
- `fre.engine.hybrid.step_rate_mixed` — local Euler ODE fallback

## Data and calibration

- `fre.io.connectome.load_connectome` / `erdos_renyi_graph` — Feather first
- `fre.calibration.apply_shiu_weights` / `calibrate_alpha` — `W_syn = 0.275 mV`
- `fre.validation.validate_single` / `validate_circuit` / `validate_hopkins_flywire`

## Embodiment (DD-5)

- `fre.adapters.embodied.EmbodiedEnv` / `OpenLoopStimEnv`
- `fre.adapters.flygym.FREFlyGymEnv`

CLI: `fre fit` (default model `adexp`), `fre simulate`, `fre validate --pysr-exam`.
