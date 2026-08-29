"""Console entry point: `clif-validate` == `python -m src.eval.clif_validate` in-repo.

The full ceremony surface lives in the vendored module: --release-id (replay
rejected), --signing-key-file, --access-log-key-file (fail-closed, no fallback),
--approved draft/release two-step, policy-aligned default output paths.
"""

from clif_validate._vendor.eval.clif_validate import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
