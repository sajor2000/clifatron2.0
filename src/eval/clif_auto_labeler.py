"""CLIF 2.1 auto-labeler — derive clinical outcomes from standard tables.

No manual annotation. Every outcome is computable from standard CLIF fields
so each external site auto-labels from its own tables. This is the mechanism
that makes the zero-shot federation validation work.

Outcomes (from configs/train.yaml finetune.tasks):
  in_hospital_mortality  — discharge_category == "Expired"
  new_imv_24h            — new IMV within 24h of ICU admission
  new_vasopressor_24h    — new vasopressor within 24h of ICU admission
  aki_kdigo_48h          — creatinine rise meeting KDIGO stage 1 within 48h
  new_rrt_48h            — new CRRT within 48h of ICU admission
  sepsis_48h             — suspected infection + SOFA rise within 48h
  resp_failure_48h       — P/F ratio < 300 within 48h

Usage:
    python -m src.eval.clif_auto_labeler \
        --data /path/to/clif_tables/ \
        --out labels.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def label_in_hospital_mortality(con, hospitalization_ids):
    """Expired at discharge."""
    result = con.execute(f"""
        SELECT hospitalization_id,
               CASE WHEN discharge_category = 'Expired' THEN 1 ELSE 0 END AS label
        FROM read_parquet('{_path('clif_hospitalization')}')
    """).pl()
    return result


def label_new_imv_24h(con, hospitalization_ids):
    """First IMV within 24h of first ICU admission.

    Uses clif_respiratory_support.device_category == 'IMV' and
    clif_adt for ICU admission timestamp.
    """
    result = con.execute(f"""
        WITH icu_start AS (
            SELECT hospitalization_id, MIN(in_dttm) AS icu_admit
            FROM read_parquet('{_path('clif_adt')}')
            WHERE location_category = 'icu'
            GROUP BY hospitalization_id
        ),
        first_imv AS (
            SELECT rs.hospitalization_id, MIN(rs.recorded_dttm) AS imv_time
            FROM read_parquet('{_path('clif_respiratory_support')}') rs
            WHERE rs.device_category = 'IMV'
            GROUP BY rs.hospitalization_id
        )
        SELECT i.hospitalization_id,
               CASE WHEN f.imv_time IS NOT NULL
                    AND f.imv_time <= i.icu_admit + INTERVAL '24 hours'
                    THEN 1 ELSE 0 END AS label
        FROM icu_start i
        LEFT JOIN first_imv f ON i.hospitalization_id = f.hospitalization_id
    """).pl()
    return result


def label_new_vasopressor_24h(con, hospitalization_ids):
    """New vasopressor (norepinephrine, phenylephrine, vasopressin, epinephrine,
    dopamine, dobutamine) within 24h of ICU admission."""
    vaso_list = "'norepinephrine','phenylephrine','vasopressin','epinephrine','dopamine','dobutamine'"
    result = con.execute(f"""
        WITH icu_start AS (
            SELECT hospitalization_id, MIN(in_dttm) AS icu_admit
            FROM read_parquet('{_path('clif_adt')}')
            WHERE location_category = 'icu'
            GROUP BY hospitalization_id
        ),
        first_vaso AS (
            SELECT m.hospitalization_id, MIN(m.admin_dttm) AS vaso_time
            FROM read_parquet('{_path('clif_medication_admin_continuous')}') m
            WHERE m.med_category IN ({vaso_list})
            GROUP BY m.hospitalization_id
        )
        SELECT i.hospitalization_id,
               CASE WHEN f.vaso_time IS NOT NULL
                    AND f.vaso_time <= i.icu_admit + INTERVAL '24 hours'
                    THEN 1 ELSE 0 END AS label
        FROM icu_start i
        LEFT JOIN first_vaso f ON i.hospitalization_id = f.hospitalization_id
    """).pl()
    return result


def label_aki_kdigo_48h(con, hospitalization_ids):
    """KDIGO stage 1 within 48h of ICU admission: creatinine rise ≥ 0.3 mg/dL
    within 48h or ≥ 1.5× baseline.

    baseline = first creatinine within 48h before or 1h after ICU admission.
    """
    result = con.execute(f"""
        WITH icu_start AS (
            SELECT hospitalization_id, MIN(in_dttm) AS icu_admit
            FROM read_parquet('{_path('clif_adt')}')
            WHERE location_category = 'icu'
            GROUP BY hospitalization_id
        ),
        creatinine AS (
            SELECT l.hospitalization_id, l.lab_result_dttm AS dttm,
                   l.lab_value_numeric AS val
            FROM read_parquet('{_path('clif_labs')}') l
            WHERE l.lab_category = 'creatinine' AND l.lab_value_numeric IS NOT NULL
        ),
        baseline AS (
            SELECT c.hospitalization_id,
                   FIRST(c.val ORDER BY c.dttm) AS base_creat
            FROM creatinine c
            JOIN icu_start i ON c.hospitalization_id = i.hospitalization_id
            WHERE c.dttm >= i.icu_admit - INTERVAL '48 hours'
              AND c.dttm <= i.icu_admit + INTERVAL '1 hour'
            GROUP BY c.hospitalization_id
        )
        SELECT i.hospitalization_id,
               CASE WHEN EXISTS (
                   SELECT 1 FROM creatinine c2
                   WHERE c2.hospitalization_id = i.hospitalization_id
                     AND c2.dttm > i.icu_admit
                     AND c2.dttm <= i.icu_admit + INTERVAL '48 hours'
                     AND (c2.val - b.base_creat >= 0.3
                          OR c2.val >= 1.5 * b.base_creat)
               ) THEN 1 ELSE 0 END AS label
        FROM icu_start i
        LEFT JOIN baseline b ON i.hospitalization_id = b.hospitalization_id
        WHERE b.base_creat IS NOT NULL
    """).pl()
    return result


def label_new_rrt_48h(con, hospitalization_ids):
    """New CRRT within 48h of ICU admission."""
    result = con.execute(f"""
        WITH icu_start AS (
            SELECT hospitalization_id, MIN(in_dttm) AS icu_admit
            FROM read_parquet('{_path('clif_adt')}')
            WHERE location_category = 'icu'
            GROUP BY hospitalization_id
        ),
        first_crrt AS (
            SELECT c.hospitalization_id, MIN(c.recorded_dttm) AS crrt_time
            FROM read_parquet('{_path('clif_crrt_therapy')}') c
            GROUP BY c.hospitalization_id
        )
        SELECT i.hospitalization_id,
               CASE WHEN f.crrt_time IS NOT NULL
                    AND f.crrt_time <= i.icu_admit + INTERVAL '48 hours'
                    THEN 1 ELSE 0 END AS label
        FROM icu_start i
        LEFT JOIN first_crrt f ON i.hospitalization_id = f.hospitalization_id
    """).pl()
    return result


def label_resp_failure_48h(con, hospitalization_ids):
    """P/F ratio < 300 (Berlin mild ARDS) within 48h of ICU admission.

    Uses clif_respiratory_support.fio2_set and clif_labs.po2_arterial.
    P/F = paO2 / FiO2. P/F < 300 counts as respiratory failure.
    """
    result = con.execute(f"""
        WITH icu_start AS (
            SELECT hospitalization_id, MIN(in_dttm) AS icu_admit
            FROM read_parquet('{_path('clif_adt')}')
            WHERE location_category = 'icu'
            GROUP BY hospitalization_id
        ),
        pf_events AS (
            SELECT rs.hospitalization_id, rs.recorded_dttm,
                   rs.fio2_set, l.lab_value_numeric AS pao2,
                   l.lab_value_numeric / NULLIF(rs.fio2_set, 0) AS pf_ratio
            FROM read_parquet('{_path('clif_respiratory_support')}') rs
            JOIN read_parquet('{_path('clif_labs')}') l
              ON rs.hospitalization_id = l.hospitalization_id
             AND ABS(EPOCH(rs.recorded_dttm - l.lab_result_dttm)) < 3600
            WHERE l.lab_category = 'po2_arterial'
              AND rs.fio2_set IS NOT NULL AND rs.fio2_set > 0
              AND l.lab_value_numeric IS NOT NULL
        )
        SELECT i.hospitalization_id,
               CASE WHEN EXISTS (
                   SELECT 1 FROM pf_events pf
                   WHERE pf.hospitalization_id = i.hospitalization_id
                     AND pf.recorded_dttm > i.icu_admit
                     AND pf.recorded_dttm <= i.icu_admit + INTERVAL '48 hours'
                     AND pf.pf_ratio < 300
               ) THEN 1 ELSE 0 END AS label
        FROM icu_start i
    """).pl()
    return result


OUTCOMES = {
    "in_hospital_mortality": label_in_hospital_mortality,
    "new_imv_24h": label_new_imv_24h,
    "new_vasopressor_24h": label_new_vasopressor_24h,
    "aki_kdigo_48h": label_aki_kdigo_48h,
    "new_rrt_48h": label_new_rrt_48h,
    "resp_failure_48h": label_resp_failure_48h,
}


def _path(table: str) -> Path:
    return BASE / f"{table}.parquet"


BASE = Path(".")


def auto_label(data_dir: str, outcomes: list[str] | None = None):
    """Derive labels for all requested outcomes from CLIF 2.1 tables.

    Args:
        data_dir: path to directory containing CLIF 2.1 parquet files
        outcomes: list of outcome names; None = all

    Returns:
        polars DataFrame with hospitalization_id plus one column per outcome.
    """
    import polars as pl

    global BASE
    BASE = Path(data_dir)

    con = duckdb.connect()
    outcomes = outcomes or list(OUTCOMES)
    dfs = []

    for name in outcomes:
        label_fn = OUTCOMES[name]
        try:
            df = label_fn(con, None)
            df = df.rename({"label": name})
            dfs.append(df)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, on="hospitalization_id", how="full")
    return result.fill_null(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing CLIF 2.1 parquet files")
    ap.add_argument("--out", default="labels.parquet")
    ap.add_argument("--outcomes", nargs="*",
                    help="subset of outcomes to label (default: all)")
    args = ap.parse_args()

    labels = auto_label(args.data, args.outcomes or None)
    labels.write_parquet(args.out)
    print(f"Wrote {len(labels):,} labeled stays to {args.out}")
    for col in labels.columns:
        if col != "hospitalization_id":
            pos = labels[col].sum()
            print(f"  {col}: {pos} positive ({pos/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    main()