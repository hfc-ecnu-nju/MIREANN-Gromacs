# GROMACS–REANN / MIREANN

An **unofficial research extension of GROMACS 2026.4** for REANN neural potentials,
MIREANN electrostatic embedding with geometry-dependent QM charges, real-atom
pseudobond boundaries, and a fixed-coupling MM ↔ ML/MM dual Hamiltonian.

**[中文说明、编译和运行](README_REANN_ZH.md)** ·
[Energy and force definitions](docs/mireann-methods.md) ·
[Dual Hamiltonian](docs/mireann-dual-hamiltonian.md) ·
[Validation and limitations](docs/release-validation-20261102.md) ·
[Changes](CHANGELOG_MIREANN.md)

The single-endpoint CUDA path supports GPU inference, nonbonded/PME calculations
and GPU integration with host staging. It is **not fully GPU resident**. The
new dual-Hamiltonian path uses two local native force evaluators and one CPU
integrator; its qualified tests currently cover CPU double precision only.
Both endpoints retain independent charges and electrostatic calculations.

This repository defines a particular local electrostatic replacement; it is
not a generic periodic ab initio QM/MM implementation. Reported force agreement
or endpoint recovery does not establish long-time NVE conservation. Prior
50 ps smooth-selection MECL tests still showed systematic energy drift. Read
the validation record before choosing a protocol.

The 2026-11-02 review also reproduced and fixed loss of the once-compensated
MM charges after native atom-data reconstruction. See the
[charge-persistence diagnosis](docs/mm-charge-persistence-fix-20261102.md).
Historical endpoint agreement alone did not detect this shared-path defect.

The complete upstream source, `README`, `COPYING`, `AUTHORS`, `INSTALL` and
`CITATION.cff` are retained. Changes carry `GMX_*` comments; replaced statements
are retained as `ORIGINAL` comments or in the original branch, with additional
pre-update snapshots recorded for this delivery. GROMACS is LGPL-2.1-or-later;
see `COPYING`. Models, datasets and other third-party material retain their own
licenses. Their presence on a local machine is not permission to redistribute
them. Local model files and simulation outputs are ignored by the new Git rules.

Run portable, model-independent checks from the repository root:

```bash
python3 tests/mireann/run_checks.py
```

This checks core formulas, switching derivatives, neighbor-cache membership and
dual-endpoint interpolation/IPC. It does not replace full-system, model-specific
CPU/GPU or molecular-dynamics validation. No GitHub publication is performed by
the source preparation scripts.
