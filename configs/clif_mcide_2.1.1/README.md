# CLIF 2.1 mCIDE snapshot

Public CLIF specification files, copied unchanged. They contain no patient data.

| Files | Source | Version |
|---|---|---|
| `mCIDE/**` (category CSVs), `ddl/CLIF2.1_MYSQL_ddl.sql` | https://github.com/Common-Longitudinal-ICU-data-Format/CLIF | tag `v2.1.1`, commit `356e3f3de5e1c8b687044956d6be4988a6f365dd` (2026-01-02) |
| `mCIDE/2_2_0_WIP/ecmo_mcs/*` | same tag (work-in-progress ECMO mCIDE; reference only, not enforced) | same |
| `clifpy_2.1.0/ecmo_mcs_schema.yaml` | https://github.com/Common-Longitudinal-ICU-data-Format/clifpy `clifpy/schemas/2.1/ecmo_mcs_schema.yaml` | release `v2.1.0` (2026-08-20) |

`manifest.yaml` says which file holds the permissible values of each checked
`table.column`. `src/data/clif_conformance.py` reads it, hashes every file in this
directory (the snapshot hash recorded in each vocabulary's `harmonization` record), and
refuses any category value that is neither permissible nor aliased / declared per site in
`configs/data.yaml` (`site_harmonization`).

ECMO/MCS: the CLIF 2.1 DDL defines `ecmo_configuration_category`, `control_parameter_*`,
`sweep_set` and `fdO2_set`; clifpy 2.1 defines `device_metric_name`, `device_rate`,
`sweep` and `fdO2`. Staged MIMIC follows clifpy, so `configs/data.yaml` pins the
`clifpy_2.1` variant and its permissible lists.

To update: copy the new tag's files, update `version`/`tag`/`commit` in `manifest.yaml`
and here, and rebuild every vocabulary (the snapshot hash changes).
