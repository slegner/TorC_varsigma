"""Evidence comparison for the Lambda runs against the no-lambda baseline.

    python scripts/compare_lam_evidence.py

Reports ln Z, Delta ln Z and the Bayes factor for every chain it can find,
plus posterior summaries.  Chains made under DIFFERENT priors on the shared
parameters are listed separately and never differenced -- in particular
chains/torc_ns_julia_1000_powerlaw.npz predates the stigma sampling-bound
fix in torc/priors.py (it floored logL above stigma = 3) and is reported
for provenance only.
"""
import os

import numpy as np
from anesthetic import NestedSamples

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(HERE, "..", "chains")

# label -> (filename, comparable-group)
# group "post-fix" chains share identical priors on the shared parameters and
# may be differenced against one another.
RUNS = [
    ("no-lambda  (4D: om, h0rd, s, OmL)",      "torc_ns_julia_1000_powerlaw_nostigmawall.npz", "post-fix"),
    ("lambda+OmL (5D: om, h0rd, s, OmL, lam)", "torc_ns_julia_1000_lam_powerlaw.npz",          "post-fix"),
    ("lambda only(4D: om, h0rd, s, lam)",      "torc_ns_julia_1000_lam_noOmL_powerlaw.npz",    "post-fix"),
    ("LCDM      (2D: om, h0rd)",               "lcdm_ns_1000.npz",                             "lcdm"),
    ("no-lambda, PRE-FIX (stigma walled at 3)", "torc_ns_julia_1000_powerlaw.npz",             "pre-fix"),
]

_PARAMS = ["omega_m", "h0rd", "stigma", "OmL", "lam"]


def load(fname):
    path = os.path.join(CHAINS, fname)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    ps = [p for p in _PARAMS if p in d.files]
    s = NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                      logL_birth=d["logL_birth"])
    return d, s, ps


def main():
    rows = []
    for label, fname, group in RUNS:
        got = load(fname)
        if got is None:
            print(f"[missing] {label:42s} {fname}")
            continue
        d, s, ps = got
        lz = s.logZ(1000)
        rows.append(dict(label=label, group=group, file=fname, s=s, ps=ps,
                         lnZ=float(lz.mean()), err=float(lz.std()),
                         DKL=float(s.D_KL()),
                         t=float(d["run_time"]) if "run_time" in d.files else np.nan))

    print(f"\n{'run':44s} {'ln Z':>10} {'+-':>7} {'D_KL':>7} {'time/s':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['label']:44s} {r['lnZ']:>10.4f} {r['err']:>7.4f} "
              f"{r['DKL']:>7.2f} {r['t']:>8.0f}")

    base = next((r for r in rows if r["group"] == "post-fix"
                 and r["label"].startswith("no-lambda ")), None)
    if base is not None:
        print(f"\nagainst the no-lambda baseline ({base['lnZ']:.4f} +- {base['err']:.4f}):")
        print(f"{'run':44s} {'D ln Z':>10} {'+-':>7}  verdict")
        print("-" * 80)
        for r in rows:
            if r["group"] != "post-fix" or r is base:
                continue
            d = r["lnZ"] - base["lnZ"]
            e = np.hypot(r["err"], base["err"])
            if abs(d) < 1:
                verdict = "inconclusive (|D ln Z| < 1, Jeffreys)"
            elif abs(d) < 2.5:
                verdict = "weak"
            elif abs(d) < 5:
                verdict = "moderate"
            else:
                verdict = "strong"
            fav = "lambda" if d > 0 else "no-lambda"
            print(f"{r['label']:44s} {d:>+10.4f} {e:>7.4f}  {verdict}"
                  f"{'' if abs(d) < 1 else f', favours {fav} ({np.exp(abs(d)):.2f}:1)'}")

    for r in rows:
        print(f"\n{r['label']}  [{r['file']}]")
        for p in r["ps"]:
            v = r["s"][p]
            print(f"   {p:8s} = {v.mean():11.4f} +- {v.std():.4f}"
                  f"   (68%: {v.quantile(0.16):.4f} .. {v.quantile(0.84):.4f})")
        if "lam" in r["ps"]:
            lam = np.asarray(r["s"]["lam"])
            w = np.asarray(r["s"].get_weights())
            w = w / w.sum()
            frac_low = float(w[lam < 0.05].sum())
            q95 = float(r["s"]["lam"].quantile(0.95))
            print(f"   -> posterior mass at lam < 0.05: {100*frac_low:.1f}%"
                  f"   95th pct: {q95:.4f}")
            print("      (mass piling up at 0 = the data do not want the term;"
                  " the prior reaches 0, so this is meaningful)")


if __name__ == "__main__":
    main()
