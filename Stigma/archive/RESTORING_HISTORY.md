# Stigma git history (pre-absorb)

Stigma was its own git repository until Aug 2026, when it was absorbed into the
Qtorsion repo. Its full history — `main` plus the unmerged `worktree-cpl-desi`
branch — is preserved in `stigma-git-history.bundle`.

To inspect or recover it:

```
git clone stigma-git-history.bundle stigma-old
cd stigma-old && git log --all --oneline
```

## About the cpl-desi branch

`worktree-cpl-desi` held 6 commits (CPL/w0waCDM nested-sampling fit to DESI DR2,
fgivenx functional posteriors, TorC-convention comparison). **That work is not
lost and was not copied here** — it graduated to its live home at
`../../cpl/` (PhD2025-26/distances/TorC/cpl/), which is a strict superset: same files, paths
rewritten for the new location, plus `cpl_prior_posterior.py` and further
outputs. Use TorC/cpl; the bundle only preserves the superseded commit history.
