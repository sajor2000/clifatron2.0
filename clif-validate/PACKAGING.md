# clif-validate packaging & offline install (U11)

`clif-validate` is meant to run at a CLIF site with **no outbound network** and a
**governed, signed** model bundle. This document is the reproducible procedure for
building an offline install and verifying it fails closed. The committed artifacts are:

- `uv.lock` — universal, fully-pinned resolution of every dependency (`uv lock`).
- `SBOM.json` — CycloneDX 1.6 software bill of materials generated from the lock.
- `tests/test_packaging.py` — cheap consistency guards: the lock pins every declared
  dependency, the `cryptography` floor set in review (`>=44.0.1`) is honoured by the
  pinned version, and the SBOM is valid CycloneDX listing `cryptography`.

The self-containment of the wheel (no `from src.` leaks, vendored tree matches sources)
is guarded separately by `tests/test_bundle_compatibility.py`.

## Regenerate the lock and SBOM (run after any dependency change)

```bash
cd clif-validate
uv lock                                   # refresh uv.lock
uv export --frozen --no-emit-project --format requirements-txt > /tmp/reqs.txt
uvx --from cyclonedx-bom cyclonedx-py requirements /tmp/reqs.txt \
    --output-format JSON -o SBOM.json     # refresh SBOM.json
uv run --with pytest pytest tests/test_packaging.py -q   # guard consistency
```

`test_packaging.py` fails if you change a dependency and forget to re-run these — the
same discipline the vendor-drift guard enforces for the vendored source.

## Build the offline wheelhouse (build host, WITH network)

Target platform is Linux x86_64 / CPython 3.11 (see `pyproject.toml`). Torch is the CPU
wheel unless the site installs CUDA torch first.

```bash
cd clif-validate
# 1. Build the clif-validate wheel itself.
uv build --wheel -o wheelhouse

# 2. Download every locked dependency wheel for the target platform into the wheelhouse.
uv export --frozen --no-emit-project --format requirements-txt > wheelhouse/requirements.txt
uv pip download -r wheelhouse/requirements.txt \
    --python-version 3.11 --platform manylinux2014_x86_64 --only-binary=:all: \
    -d wheelhouse

# 3. Ship `wheelhouse/` (wheels + requirements.txt) plus SBOM.json to the site out of band.
```

## Install offline and verify fail-closed (site host, NO network)

```bash
# Install ONLY from the wheelhouse; --no-index forbids any network index.
python -m venv /opt/clif-validate-venv
/opt/clif-validate-venv/bin/pip install --no-index --find-links wheelhouse clif-validate

# Smoke: prove the install actually loads a bundle AND fails closed. --help alone would
# pass even if fail-closed bundle loading were broken, so point it at an UNSIGNED bundle
# with no trust root and assert it refuses (exercises bundle load + signature gate).
CLIF_OUT=$(/opt/clif-validate-venv/bin/clif-validate \
    --checkpoint <unsigned_bundle_dir> --data <clif_tables> \
    --episode-artifact <episodes.parquet> --site-id SMOKE \
    --release-id smoke --rollback-state /tmp/smoke_rollback.json 2>&1 || true)
echo "$CLIF_OUT" | grep -qiE 'unsigned|signature|trust' \
    || { echo "FAIL: unsigned bundle was not refused"; exit 1; }
echo "OK: install loads a bundle and fails closed on an unsigned one"

# Governed run against a signed bundle + the site's out-of-band trust root.
# A release must be signature-verified, content-hash-bound, and rollback-protected:
/opt/clif-validate-venv/bin/clif-validate \
    --checkpoint <signed_bundle_dir> --data <clif_tables> \
    --episode-artifact <episodes.parquet> --site-id <SITE> \
    --release-id <REL> --trust-roles <trust_roles.yaml> \
    --rollback-state <site_local_rollback.json> \
    --signing-key-file <site_report_key> --access-log-key-file <site_chain_key>
# (writes a DRAFT + <out>.draft.sha256; re-run with --approved --approved-hash <hash> to release)
```

### Fail-closed checks the offline install must pass (manual/CI acceptance)

Run these on the site host to confirm the trust layer holds with no network:

1. **No network at runtime.** Under a network namespace with no route (or
   `firejail --net=none` / an egress-blocked container), a full validate → draft →
   approve cycle completes. Any outbound connection attempt is a failure.
2. **Unsigned bundle refused.** A bundle with `bundle_manifest.sig` removed fails to load
   (`unsigned ... governed bundle must carry a releaser signature`).
3. **Untrusted/revoked signature refused.** A bundle signed by a key not in — or revoked
   by — the trust root fails closed.
4. **Approval bound to content.** `--approved` without `--approved-hash` is refused; an
   `--approved-hash` that does not match the reviewed draft is refused.
5. **No `--allow-unsigned` release.** `--allow-unsigned --approved` is refused (a
   not-for-release bundle can never publish a governed report).

## Not automated here (governance / ops — recorded, not faked)

These need a real build host, a signing HSM, and a site deployment; they are acceptance
criteria for a release, not cheap unit tests:

- Building the wheelhouse and downloading platform wheels (needs network + the target
  platform; the SBOM/lock guards above are the committed, testable slice).
- The runtime network-egress block itself (a host/container control, not package code).
- Real release-signing key custody (HSM) and out-of-band trust-root distribution — see
  `configs/trust_roles.yaml` and its `pending_governance` block.
