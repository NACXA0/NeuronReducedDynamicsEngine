from __future__ import annotations

from pathlib import Path

from fre.offline.chirp import chirp_impedance
from fre.offline.fit import attach_lut, fit_fi, fit_spike_lut, save_fit
from fre.offline.pysr_backend import admission_exam
from fre.offline.srm import attach_srm, fit_srm_kernels
from fre.validation import validate_single


def main() -> None:
    adexp = fit_fi(
        "adexp",
        type_id="aCC",
        I_min=0.0,
        I_max=0.8,
        n_I=12,
        t_total=500.0,
        window=300.0,
        dt=0.05,
        chirp=False,
    )
    z = chirp_impedance("adexp", freqs_hz=[2.0, 8.0, 20.0, 40.0], cycles=3.0, dt=0.1)
    print(
        f"AdExp R²={adexp.r2:.4f} layer={adexp.layer} quality={adexp.quality} "
        f"|Z| peak={z.has_peak} {z.notes}"
    )
    report = validate_single(adexp, t_total=400.0, window=250.0)
    print(f"validate_single AdExp nfr4_pass={report.nfr4_pass} mse={report.mse:.3f}")
    adexp = attach_srm(adexp, fit_srm_kernels("adexp", t_kernel=40.0, dt=0.1))
    lut = fit_spike_lut("adexp", I_min=0.0, I_max=0.8, n_I=8, n_dt=8)
    adexp = attach_lut(adexp, lut)
    Path("artifacts").mkdir(exist_ok=True)
    save_fit(adexp, "artifacts/adexp.npz")
    exam = admission_exam()
    print(f"PySR admission: {exam.status} ({exam.reason})")
    print("wrote artifacts/adexp.npz")


if __name__ == "__main__":
    main()
