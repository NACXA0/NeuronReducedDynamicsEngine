# Changelog

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
