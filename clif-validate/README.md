# clif-validate

Turnkey federated validation for [CLIF](https://clif-consortium.github.io/) sites.
A site receives a **frozen, hash-sealed model bundle** and this package; it runs the
bundle on its **local** CLIF 2.1 parquet tables, and only **disclosure-controlled
aggregate metrics** leave the node. No raw rows, no labels, no gradients, no model
updates cross the institutional boundary.

The package, its source, and the bundle contract are open (MIT). Trained model
weights are **governed artifacts** distributed separately under data-use agreements —
this package verifies and runs a bundle; it does not contain one.

## What a site runs

```bash
pip install clif-validate   # Linux x86_64 / Python 3.11+ (POSIX-only: uses fcntl)

clif-validate \
  --checkpoint /path/to/bundle \
  --data /path/to/local_clif_parquet/ \
  --episode-artifact /path/to/episodes.parquet \
  --site-id SITE-07 \
  --release-id 2026-08-29-site07-v0 \
  --signing-key-file /secure/site07.key \
  --access-log-key-file /secure/site07-accesslog.key \
  --out output/final_no_phi/site_07.json
```

Without `--approved` the run writes a **local draft** (`<out>.draft`, unsigned,
unledgered) and stops, so the site's disclosure review sees exactly what would
leave. Re-running with `--approved` stamps, signs, records the access, and
releases through a write-ahead disclosure ledger. Replaying a `--release-id` is
rejected; a missing access-log key fails the run *before* anything is published.

## What the bundle contains

`bundle_manifest.json` (identity hashes + per-file SHA-256 + per-outcome zero-shot
query parameters), the frozen backbone + trained heads, the pinned vocabulary and
numeric bin edges, the resolved data config, the frozen outcome contract, and the
**artifact policy** the run enforces — every file hash-covered, and a file the
manifest does not list is a hard failure.

## Self-test without any data

The vendored synthetic fixture exercises the entire pipeline — tokenization,
zero-shot inference, disclosure suppression, signing, ledger — on generated
arithmetic "patients":

```python
from clif_validate._vendor.eval.synthetic_bundle import (
    build_synthetic_site, build_synthetic_bundle,
)
```

A green synthetic run proves the machinery (and only the machinery).

## Provenance of this code

Everything under `clif_validate/_vendor/` is a byte-identical, import-rewritten
copy of the [CLIFATRON repository](https://github.com/sajor2000/clifatron2.0)'s
`src/` modules, produced by `scripts/sync_vendor.py` and guarded by
`vendor_manifest.json` — the repo and the wheel run the same implementation, and
drift on either side fails a test. Do not edit vendored files by hand.

## License

MIT. Model weights are not part of this package and are governed separately.
