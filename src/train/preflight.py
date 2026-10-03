"""L40 pre-flight: one command run on the training node before the first training run
(plan U19; KTD3, KTD6, KTD12).

It prints a pass/fail table and exits non-zero on any failure. Every line is aggregate:
counts, sizes, hashes (shortened), versions, check names. It never prints a row, an
identifier or a timestamp of patient data.

    uv run python -m src.train.preflight \\
        --episodes mimic=output/intermediate_phi/episodes.parquet \\
        --split-freeze output/final_no_phi/split_freeze.json

Checks, in order:

- environment: torch / CUDA / cuDNN / driver versions, git commit, and a dirty-tree refusal
  (``--allow-dirty`` turns it into a warning);
- GPU: CUDA available, two devices, native bf16, a usable driver (``nvidia-smi`` runs; the
  node's known driver/library mismatch is reported as such), free memory per device, and a
  two-rank smoke step: two processes (``torch.multiprocessing.spawn``) join an nccl process
  group through ``engine.setup_ddp``, train a tiny ``pretrain.Model`` for two accumulated
  updates on synthetic batches through ``engine.train`` under DDP and bf16, and must end
  with bit-identical parameters. Without CUDA every GPU check FAILS, unless ``--skip-gpu``
  (a Mac rehearsal): the GPU checks are then skipped and the same smoke runs over gloo on
  CPU (the ``--allow-cpu-ddp`` path);
- data binding, per tokenization arm of ``configs/experiment_matrix.yaml`` and per site:
  the vocabulary is tokenizer v2 and not fit on a sample, every site's vocab.json and every
  shard row's ``artifact_hashes`` equal the arm's vocabulary and segments binding
  (``segments.compare_binding``), the arm's representation matches its vocabulary
  (``run_tokenization_ablation.check_arm_vocab``), and a soft arm's shard carries soft bins;
- value stats: present, bound to the arm's vocabulary and segments, fit on ``train``, and
  covering every numeric token of a sample of the shard's train stays (stats fit on the
  24-hour shard miss tokens seen only outside the ICU window);
- split freeze (KTD6): ``split_freeze.json`` must exist and match every site's episode
  artifact (content split hash, file hash, configured proportions and seed), and every
  shard stay's partition must equal the episode artifact's. ``--write-split-freeze``
  writes the record once the held-out share is decided;
- thresholds: ``configs/thresholds.yaml`` loads; the edge-distance check over every arm's
  frozen vocabulary (``run_matrix.edge_distance_table``) fails, naming each refused
  control and arm; competing-risk causes still ``status: proposed`` are a warning;
- memory: the ``GemCorpus`` cache is built and memory-mapped for a sample of the shard
  (a copy under the governed scratch directory; the shard and its own cache are not
  touched), bytes per event are measured, and per-rank and per-node memory for the full
  train + validation corpus are projected against available RAM (warn above 70 %, fail
  above 90 %); the cache directory must be on a local filesystem (``fcntl`` locks and
  memory maps are not safe on NFS/SMB);
- schedule: the optimizer updates each matrix budget gives (the launch's own sampler and
  `engine.resolve_total_steps`), and the warm-up and checkpoint interval
  `engine.resolve_schedule` gives each run — the same rules the launcher applies: fails
  when a run would never leave warm-up (an absolute ``warmup_steps`` that outgrows it) or
  write no checkpoint at all; warns when it writes only the final checkpoint (an
  absolute ``ckpt_every`` longer than the run) or gets no warm-up;
- disk: free space for the cache still to be built and for checkpoints.

``--synthetic`` builds a synthetic site in a temporary directory and runs every non-GPU
check on it (with ``--skip-gpu``: the gloo rehearsal too).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs/experiment_matrix.yaml"
ABLATION_PATH = ROOT / "configs/tokenization_ablation.yaml"
TRAIN_CONFIG_PATH = ROOT / "configs/train.yaml"
DATA_CONFIG_PATH = ROOT / "configs/data.yaml"
POLICY_PATH = ROOT / "configs/artifact_policy.yaml"
THRESHOLDS_PATH = ROOT / "configs/thresholds.yaml"
DEFAULT_SPLIT_FREEZE = Path("output/final_no_phi/split_freeze.json")
DEFAULT_EPISODES = Path("output/intermediate_phi/episodes.parquet")
DEFAULT_SCRATCH = Path("output/intermediate_phi/preflight_scratch")
GEM_EVENTS = "gem_events.parquet"
SPLIT_FREEZE_SCHEMA = "1.0.0"
EXPECTED_GPUS = 2
MEMORY_WARN, MEMORY_FAIL = 0.70, 0.90
SMOKE_TIMEOUT_S = 300
PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
# Filesystems on which the cache's fcntl lock or its memory maps are not trustworthy.
NETWORK_FS = {"nfs", "nfs4", "cifs", "smb", "smb2", "smb3", "smbfs", "afpfs", "webdav",
              "ncpfs", "9p", "lustre", "gpfs", "beegfs", "ceph", "glusterfs", "ocfs2",
              "gfs2", "fuse.sshfs", "fuse.s3fs", "fuse.gcsfuse", "fuse.rclone",
              "fuse.glusterfs", "fuse.cephfs", "afs", "panfs", "orangefs", "pvfs2"}
DRIVER_MISMATCH = re.compile(r"driver/library version mismatch|NVML library version",
                             re.IGNORECASE)


class PreflightError(RuntimeError):
    """A pre-flight input is unusable (bad flag, unreadable artifact)."""


# ------------------------------------------------------------------ report

@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> Check:
        check = Check(name, status, detail)
        self.checks.append(check)
        return check

    def extend(self, checks: Sequence[Check]) -> None:
        self.checks.extend(checks)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def table(self) -> str:
        width = max([len(c.name) for c in self.checks] + [5])
        lines = [f"{'check':<{width}}  status  detail", f"{'-' * width}  ------  ------"]
        for c in self.checks:
            detail = c.detail.replace("\n", " ")
            lines.append(f"{c.name:<{width}}  {c.status:<6}  {detail}")
        counts = {s: sum(c.status == s for c in self.checks) for s in (PASS, WARN, FAIL, SKIP)}
        lines.append(f"\n{counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail, "
                     f"{counts[SKIP]} skipped -> "
                     f"{'PRE-FLIGHT FAILED' if counts[FAIL] else 'PRE-FLIGHT PASSED'}")
        return "\n".join(lines)


def _short(value: str | None, n: int = 12) -> str:
    return "none" if not value else f"{str(value)[:n]}…"


def _gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return Path(path).name


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ environment

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def nvidia_smi() -> tuple[bool, str, str | None]:
    """(usable, detail, driver version). The node's known failure is a driver/library
    version mismatch, which only a reboot clears."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60,
            check=False)
    except FileNotFoundError:
        return False, "nvidia-smi not found (no NVIDIA driver on this machine)", None
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out", None
    text = (out.stdout + out.stderr).strip()
    if out.returncode != 0:
        if DRIVER_MISMATCH.search(text):
            return False, ("driver/library version mismatch (the known node fault): reboot "
                           "the node before training"), None
        return False, f"nvidia-smi exited {out.returncode}: {text[:160]}", None
    rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.strip()]
    driver = rows[0][1].strip() if rows and len(rows[0]) > 1 else None
    names = sorted({r[0].strip() for r in rows})
    return True, f"{len(rows)} GPU(s) {', '.join(names)}; driver {driver}", driver


def environment_checks(*, allow_dirty: bool) -> list[Check]:
    import torch

    checks = []
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    checks.append(Check("env: versions", PASS,
                        f"python {platform.python_version()}, torch {torch.__version__}, "
                        f"CUDA build {torch.version.cuda}, cuDNN {cudnn}, "
                        f"{platform.system()} {platform.machine()}"))
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if commit is None or status is None:
        checks.append(Check("env: git tree", FAIL, "not a git checkout (cannot record the "
                            "commit every checkpoint should trace to)"))
        return checks
    dirty = [line for line in status.splitlines() if line.strip()]
    detail = f"commit {commit[:12]} on {branch}; {len(dirty)} modified tracked file(s)"
    if not dirty:
        checks.append(Check("env: git tree", PASS, detail + " (clean)"))
    elif allow_dirty:
        checks.append(Check("env: git tree", WARN, detail + " (allowed by --allow-dirty)"))
    else:
        checks.append(Check("env: git tree", FAIL, detail + ": commit or stash before "
                            "training, or pass --allow-dirty"))
    return checks


# ------------------------------------------------------------------ GPU

def gpu_checks(*, skip_gpu: bool, expected: int = EXPECTED_GPUS) -> list[Check]:
    """CUDA, device count, native bf16, driver and free memory. `skip_gpu` reports them
    as skipped (a Mac rehearsal); otherwise a missing GPU is a failure, never a crash."""
    import torch

    names = ("gpu: CUDA available", "gpu: device count", "gpu: bf16", "gpu: driver",
             "gpu: free memory")
    if skip_gpu:
        return [Check(name, SKIP, "--skip-gpu") for name in names]
    checks = []
    ok, detail, _ = nvidia_smi()
    available = torch.cuda.is_available()
    checks.append(Check(names[0], PASS if available else FAIL,
                        f"torch sees CUDA (build {torch.version.cuda})" if available else
                        "torch.cuda.is_available() is False"))
    if not available:
        checks += [Check(names[1], FAIL, "no CUDA"), Check(names[2], FAIL, "no CUDA"),
                   Check(names[3], FAIL, detail), Check(names[4], FAIL, "no CUDA")]
        return checks
    count = torch.cuda.device_count()
    checks.append(Check(names[1], PASS if count >= expected else FAIL,
                        f"{count} device(s), expected {expected}"))
    bf16 = []
    free = []
    for index in range(count):
        with torch.cuda.device(index):
            bf16.append(torch.cuda.is_bf16_supported(including_emulation=False))
        f, t = torch.cuda.mem_get_info(index)
        free.append(f"cuda:{index} {_gib(f)} free of {_gib(t)}")
    checks.append(Check(names[2], PASS if bf16 and all(bf16) else FAIL,
                        f"native bf16 on {sum(bf16)}/{count} device(s)"))
    checks.append(Check(names[3], PASS if ok else FAIL, detail))
    checks.append(Check(names[4], PASS, "; ".join(free)))
    return checks


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


SMOKE_VOCAB, SMOKE_TARGETS, SMOKE_VALUE_BINS, SMOKE_TIME_BINS = 32, 2, 5, 4
SMOKE_BINDING = {"tokenizer_version": "2", "vocabulary": "0" * 64, "numeric_edges": "1" * 64}
SMOKE_BATCHES, SMOKE_ACCUM, SMOKE_UPDATES = 8, 2, 2   # 4 microbatches per rank


def _smoke_mcfg() -> dict:
    return {
        "trunk": {"d_model": 32, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                  "rope_base": 10000.0, "tied_embeddings": False},
        "heads": {
            "next_event": {"enabled": True, "weight": 0.2},
            "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": SMOKE_TIME_BINS},
            "threshold_hazard": {"enabled": True, "weight": 1.0,
                                 "n_time_bins": SMOKE_TIME_BINS, "threshold_embed_dim": 4},
            "value_regression": {"enabled": True, "weight": 0.5},
        },
        "compile": False,
    }


def _smoke_batch(seed: int, *, n: int = 2, t: int = 16) -> dict:
    """One collated microbatch in the engine's batch contract (synthetic)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    token = torch.randint(1, SMOKE_VOCAB, (n, t), generator=g)
    ntp_mask = torch.ones(n, t, dtype=torch.bool)
    ntp_mask[:, -1] = False
    value_mask = torch.rand(n, t, generator=g) > 0.5
    value_mask[:, 0] = True
    anchor = torch.ones(n, dtype=torch.bool)
    return {
        "input_ids": token, "attention_mask": torch.ones(n, t, dtype=torch.bool),
        "pos_min": torch.arange(t).repeat(n, 1) * 5,
        "anchor_idx": torch.full((n,), t - 1, dtype=torch.long),
        "ntp_target": torch.roll(token, -1, dims=1), "ntp_mask": ntp_mask,
        "value_target": torch.randn(n, t, generator=g), "value_mask": value_mask,
        "input_value": torch.randn(n, t, generator=g), "input_value_mask": value_mask.clone(),
        "cr_mask": anchor, "cr_type": torch.randint(-1, SMOKE_TARGETS + 1, (n,), generator=g),
        "cr_bin": torch.randint(0, SMOKE_TIME_BINS, (n,), generator=g),
        "th_mask": anchor.clone(),
        "th_target": torch.randint(0, SMOKE_TARGETS, (n,), generator=g),
        "th_tau": torch.randint(0, SMOKE_VALUE_BINS, (n,), generator=g),
        "th_dir": torch.randint(0, 2, (n,), generator=g),
        "th_crossed": torch.randint(-1, SMOKE_TIME_BINS, (n,), generator=g),
        "th_observed_bin": torch.randint(1, SMOKE_TIME_BINS + 1, (n,), generator=g),
    }


class _Batches:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def _smoke_rank(local: int, port: int, out_dir: str, cpu: bool) -> None:
    """One rank of the two-rank smoke step (module top level: `mp.spawn` pickles it)."""
    os.environ.update({"RANK": str(local), "LOCAL_RANK": str(local), "WORLD_SIZE": "2",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before any CUDA call in this process
        os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from src.train.engine import TrainConfig, select_device, setup_ddp, train, wrap_ddp
    from src.train.pretrain import Model, build_optimizer, build_scheduler

    out = Path(out_dir)
    result: dict = {"rank": local}
    try:
        if cpu:
            torch.set_num_threads(1)
        local_rank, _ = setup_ddp(allow_cpu=cpu)
        backend = dist.get_backend()
        expected = "gloo" if cpu else "nccl"
        if backend != expected:
            raise RuntimeError(f"joined a {backend} process group, expected {expected}")
        device = select_device(local_rank)
        result.update(backend=backend, device=device.type)
        if device.type == "cuda":
            result["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
            probe = torch.ones(8, 8, device=device, dtype=torch.bfloat16)
            result["bf16_matmul_finite"] = bool(torch.isfinite(probe @ probe).all())
        mcfg = _smoke_mcfg()
        torch.manual_seed(0)
        model = Model(SMOKE_VOCAB, SMOKE_TARGETS, mcfg, n_value_bins=SMOKE_VALUE_BINS)
        model = wrap_ddp(model.to(device), device, local_rank)
        tcfg = {
            "optimizer": {"lr": 1e-2, "weight_decay": 0.1, "betas": [0.9, 0.95],
                          "grad_clip": 1.0},
            "schedule": {"warmup_steps": 1, "total_steps": SMOKE_UPDATES,
                         "cosine_decay": True},
            "batch": {"per_gpu": 2, "grad_accum": SMOKE_ACCUM},
            "runtime": {"precision": "bf16", "num_workers": 0, "log_every": 1000,
                        "ckpt_every": 10**9, "ckpt_dir": str(out / "ckpt")},
            "eval_schedule": {"val_every": 10**9},
        }
        opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = build_scheduler(opt, SMOKE_UPDATES, 1)
        dataset = _Batches([_smoke_batch(i) for i in range(SMOKE_BATCHES)])
        loader = DataLoader(dataset, batch_size=None,
                            sampler=DistributedSampler(dataset, shuffle=False))
        trained, manifest = train(model, loader, None, opt, scheduler,
                                  TrainConfig({}, tcfg, mcfg, SMOKE_UPDATES), device,
                                  vocab_binding=SMOKE_BINDING)
        module = trained.module if hasattr(trained, "module") else trained
        digest = hashlib.sha256()
        finite = True
        for name, param in sorted(module.named_parameters()):
            data = param.detach().float().cpu()
            finite = finite and bool(torch.isfinite(data).all())
            digest.update(name.encode())
            digest.update(data.numpy().tobytes())
        result.update(updates=int(manifest.ledger.get("optimizer_updates", 0)),
                      param_sha256=digest.hexdigest(), finite=finite, ok=True)
    except BaseException as exc:  # reported by the parent
        result.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    finally:
        (out / f"rank_{local}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


def ddp_smoke(*, cpu: bool, timeout: float = SMOKE_TIMEOUT_S) -> Check:
    """Two ranks through the real DDP path; pass = both finish their updates with finite,
    bit-identical parameters (and, on CUDA, native bf16)."""
    import torch.multiprocessing as mp

    name = "gpu: two-rank DDP smoke" if not cpu else "gpu: two-rank DDP smoke (gloo, CPU)"
    with tempfile.TemporaryDirectory(prefix="preflight_ddp_") as td:
        context = mp.spawn(_smoke_rank, args=(_free_port(), td, cpu), nprocs=2, join=False)
        deadline = time.monotonic() + timeout
        error = None
        try:
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    error = f"timed out after {timeout:.0f} s"
                    break
        except Exception as exc:  # noqa: BLE001 - a rank failed
            error = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
                process.join(5)
        results = []
        for rank in range(2):
            path = Path(td) / f"rank_{rank}.json"
            results.append(json.loads(path.read_text()) if path.exists() else {"ok": False})
    rank_errors = [r.get("error") for r in results if r.get("error")]
    if error or rank_errors or not all(r.get("ok") for r in results):
        return Check(name, FAIL, "; ".join(filter(None, [error, *rank_errors]))
                     or "a rank did not report")
    updates = [r["updates"] for r in results]
    same = results[0]["param_sha256"] == results[1]["param_sha256"]
    problems = []
    if updates != [SMOKE_UPDATES] * 2:
        problems.append(f"updates {updates}, expected {SMOKE_UPDATES} per rank")
    if not same:
        problems.append("parameters differ across ranks")
    if not all(r["finite"] for r in results):
        problems.append("non-finite parameters")
    if not cpu and not all(r.get("bf16_native") and r.get("bf16_matmul_finite")
                           for r in results):
        problems.append("bf16 not native on a rank")
    detail = (f"{results[0]['backend']} on {results[0]['device']}, {updates[0]} accumulated "
              f"updates per rank (accum {SMOKE_ACCUM}), parameters identical: {same}")
    return Check(name, FAIL if problems else PASS,
                 "; ".join(problems) + (": " if problems else "") + detail)


# ------------------------------------------------------------------ split freeze (KTD6)

def _site_paths(items: Sequence[str], default_site: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        site, sep, path = item.partition("=")
        if not sep:
            site, path = default_site, item
        if site in out:
            raise PreflightError(f"site {site!r} given twice")
        out[site] = Path(path)
    return out


def _load_train_contract(train_config: str | Path) -> dict:
    return yaml.safe_load(Path(train_config).read_text())["data_contract"]


def episode_split_summary(path: str | Path, train_config: str | Path = TRAIN_CONFIG_PATH
                          ) -> dict:
    """Aggregate identity of one episode artifact: the file's SHA-256, the content split
    hash (verified by `cohort.validate_episode_artifact`), the observed partition shares
    of eligible episodes and the configured proportions and seed."""
    import polars as pl

    from src.data.cohort import validate_episode_artifact

    episodes = pl.read_parquet(path)
    validate_episode_artifact(episodes)            # recomputes both content hashes
    eligible = episodes.filter(pl.col("eligible"))
    counts = dict(eligible.group_by("partition").len().iter_rows())
    total = max(sum(counts.values()), 1)
    contract = _load_train_contract(train_config)
    return {
        "file": Path(path).name,
        "file_sha256": file_sha256(path),
        "split_sha256": str(episodes["split_sha256"][0]),
        "episode_sha256": str(episodes["episode_sha256"][0]),
        "eligible_episodes": int(sum(counts.values())),
        "observed_shares": {k: round(v / total, 4) for k, v in sorted(counts.items())},
        "configured_proportions": dict(contract["partitions"]),
        "split_seed": int(contract["split_seed"]),
        # Item 46: the held-out stratification the split was built with (config only).
        "held_out_stratification": dict(contract.get("held_out_stratification") or {}),
    }


def held_out_arm_counts(blind_json: str | Path) -> dict:
    """Arm sizes outside the pretraining partition from a U13 blind-stage export
    (cells ``<site>|held_out|eligible|arm=<arm>``), as released (suppressed cells keep
    their status, never a number)."""
    payload = json.loads(Path(blind_json).read_text())
    if payload.get("stage") != "blind":
        raise PreflightError(f"{Path(blind_json).name} is not a blind-stage audit export")
    counts: dict[str, dict] = {}
    for key, cell in (payload.get("cells") or {}).items():
        parts = key.split("|")
        if len(parts) == 4 and parts[1] == "held_out" and parts[2] == "eligible" \
                and parts[3].startswith("arm="):
            value = cell.get("n") if cell.get("status") == "evaluable" else cell.get("status")
            counts.setdefault(parts[0], {})[parts[3][4:]] = value
    return {"release_id": payload.get("release_id"),
            "disclosure_status": payload.get("disclosure_status"),
            "file_sha256": file_sha256(blind_json), "held_out_arms": counts}


def build_split_freeze(episodes: Mapping[str, str | Path], *, approver: str,
                       date: str | None = None, audit_blind: str | Path | None = None,
                       train_config: str | Path = TRAIN_CONFIG_PATH) -> dict:
    if not approver or not approver.strip():
        raise PreflightError("--approver is required: the split freeze is a recorded decision")
    record = {
        "schema_version": SPLIT_FREEZE_SCHEMA,
        "decision": "KTD6: the split is baked into every checkpoint; it is frozen before "
                    "the first L40 training run",
        "approver": approver.strip(),
        "date": date or dt.datetime.now(dt.UTC).date().isoformat(),
        "sites": {site: episode_split_summary(path, train_config)
                  for site, path in episodes.items()},
    }
    if audit_blind is not None:
        record["audit_blind"] = held_out_arm_counts(audit_blind)
    return record


def write_split_freeze(path: str | Path, record: Mapping, *, force: bool = False) -> Path:
    """Write the freeze once. An existing freeze with a different split is refused (it
    would orphan every checkpoint trained on it) unless `force`."""
    path = Path(path)
    if path.exists() and not force:
        old = json.loads(path.read_text())
        old_hashes = {s: v.get("split_sha256") for s, v in (old.get("sites") or {}).items()}
        new_hashes = {s: v["split_sha256"] for s, v in record["sites"].items()}
        if old_hashes != new_hashes:
            raise PreflightError(f"{path} already freezes a different split; pass --force "
                                 "only if no checkpoint has been trained on the old one")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def check_split_freeze(freeze_path: str | Path | None, episodes: Mapping[str, str | Path],
                       *, train_config: str | Path = TRAIN_CONFIG_PATH) -> Check:
    name = "split: frozen hash (KTD6)"
    if freeze_path is None or not Path(freeze_path).exists():
        return Check(name, FAIL, "no split_freeze.json: decide the held-out share from the "
                     "audit blind stage, then run --write-split-freeze (refusing to train)")
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze.get("schema_version") != SPLIT_FREEZE_SCHEMA or not freeze.get("approver"):
        return Check(name, FAIL, "split_freeze.json has no schema version or approver")
    frozen = freeze.get("sites") or {}
    problems, notes = [], []
    if not episodes:
        problems.append("no --episodes artifact given to verify")
    for site, path in episodes.items():
        if site not in frozen:
            problems.append(f"site {site} is not in the freeze")
            continue
        if not Path(path).exists():
            problems.append(f"site {site}: episode artifact missing")
            continue
        try:
            now = episode_split_summary(path, train_config)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"site {site}: episode artifact invalid ({type(exc).__name__})")
            continue
        was = frozen[site]
        if now["split_sha256"] != was.get("split_sha256"):
            problems.append(f"site {site}: split hash {_short(now['split_sha256'])} != "
                            f"frozen {_short(was.get('split_sha256'))}")
        elif now["file_sha256"] != was.get("file_sha256"):
            notes.append(f"site {site}: same split content, artifact file rewritten")
        if now["configured_proportions"] != was.get("configured_proportions") \
                or now["split_seed"] != was.get("split_seed"):
            problems.append(f"site {site}: configs/train.yaml partitions or split_seed "
                            "changed since the freeze")
        if now["held_out_stratification"] != was.get("held_out_stratification", {}):
            problems.append(f"site {site}: configs/train.yaml held_out_stratification "
                            "changed since the freeze")
    if problems:
        return Check(name, FAIL, "; ".join(problems))
    shares = {s: frozen[s].get("configured_proportions") for s in episodes}
    detail = (f"frozen {freeze.get('date')} by {freeze.get('approver')}; "
              + "; ".join(f"{s} split {_short(frozen[s]['split_sha256'])}" for s in episodes)
              + f"; proportions {next(iter(shares.values()))}")
    if notes:
        return Check(name, WARN, detail + "; " + "; ".join(notes))
    return Check(name, PASS, detail)


def check_shard_partitions(site: str, shard: str | Path, episodes: str | Path) -> Check:
    """Every stay of the shard carries the episode artifact's partition (the shard was
    tokenized from the frozen split)."""
    import polars as pl

    name = f"split: {site} shard partitions"
    stays = (pl.scan_parquet(shard).select("hosp_id", "partition").unique().collect())
    if stays["hosp_id"].n_unique() != stays.height:
        return Check(name, FAIL, "a stay carries two partitions in the shard")
    ref = pl.scan_parquet(episodes).select(
        pl.col("hospitalization_id").alias("hosp_id"),
        pl.col("partition").alias("episode_partition")).collect()
    joined = stays.join(ref, on="hosp_id", how="left")
    missing = int(joined["episode_partition"].is_null().sum())
    differ = int((joined["episode_partition"] != joined["partition"]).sum())
    detail = f"{stays.height} stays; {missing} not in the episode artifact; {differ} differ"
    return Check(name, FAIL if missing or differ else PASS,
                 detail + ("" if not (missing or differ) else
                           ": re-tokenize from the frozen episode artifact"))


# ------------------------------------------------------------------ data binding

@dataclass
class ArmData:
    """One tokenization arm as the launch reads it: a shard directory per site
    (vocab.json + gem_events.parquet) and the value-stats file."""

    name: str
    sites: dict[str, Path]
    value_stats: Path
    config: dict | None = None          # configs/tokenization_ablation.yaml arms.<name>

    @property
    def vocab(self) -> Path:
        return next(iter(self.sites.values())) / "vocab.json"


def matrix_arms(*, matrix_path: str | Path = MATRIX_PATH,
                ablation_path: str | Path = ABLATION_PATH,
                overrides: Sequence[str] = (), only: Sequence[str] | None = None
                ) -> list[ArmData]:
    """The arms of the experiment matrix with their launch directories. `overrides`
    entries are ``ARM=DIR`` (first launch site) or ``ARM:SITE=DIR``."""
    matrix = yaml.safe_load(Path(matrix_path).read_text())
    abl = yaml.safe_load(Path(ablation_path).read_text())
    sites = list(matrix["launch"]["sites"])
    data = {arm: {site: Path(d) for site, d in dirs.items()}
            for arm, dirs in matrix["arm_data"].items()}
    for item in overrides:
        key, sep, path = item.partition("=")
        if not sep:
            raise PreflightError(f"--arm-dir takes ARM=DIR or ARM:SITE=DIR, got {item!r}")
        arm, _, site = key.partition(":")
        if arm not in data:
            raise PreflightError(f"--arm-dir names unknown arm {arm!r}")
        data[arm][site or sites[0]] = Path(path)
    names = list(only) if only else list(data)
    unknown = sorted(set(names) - set(data))
    if unknown:
        raise PreflightError(f"unknown arm(s) {unknown}")
    stats_file = matrix["launch"]["value_stats_file"]
    arms = []
    for name in names:
        config = {**abl["arms"][name], "name": name}
        if config.get("primary_vocab") and "clinical_soft" in data:
            # The continuous-fused arm's segments must be the primary clinical arm's as
            # this run lays it out (the config's default path is the L40 layout).
            config["primary_vocab"] = str(data["clinical_soft"][sites[0]] / "vocab.json")
        arms.append(ArmData(name, dict(data[name]), data[name][sites[0]] / stats_file,
                            config))
    return arms


def _shard_bindings(shard: Path) -> list[dict]:
    import polars as pl

    return (pl.scan_parquet(shard).select("artifact_hashes").unique().collect()
            ["artifact_hashes"].to_list())


def _soft_width(shard: Path) -> int:
    import polars as pl

    lf = pl.scan_parquet(shard)
    if "soft_token" not in lf.collect_schema().names():
        return 0
    first = lf.select(pl.col("soft_token").list.first().list.len().max()).collect().item()
    return int(first or 0)


def check_arm_binding(arm: ArmData) -> list[Check]:
    """Vocabulary, shards and segments of one arm bound to each other (module docstring)."""
    from src.data.segments import (
        artifact_binding,
        compare_binding,
        is_sample_vocab,
        load_vocab_blob,
    )

    name = f"data: {arm.name} binding"
    if not arm.vocab.exists():
        return [Check(name, FAIL, f"no vocab.json in {_rel(arm.vocab.parent)}")]
    try:
        blob = load_vocab_blob(arm.vocab)
        binding = artifact_binding(blob)
    except Exception as exc:  # noqa: BLE001
        return [Check(name, FAIL, f"vocabulary unusable: {str(exc)[:200]}")]
    checks = []
    if is_sample_vocab(blob):
        checks.append(Check(f"data: {arm.name} vocabulary", FAIL,
                            "refused: fit on a verification sample (provenance.sample: "
                            "true), smoke-only; re-tokenize the reference site's full "
                            "train partition without --sample-episodes"))
    else:
        checks.append(Check(f"data: {arm.name} vocabulary", PASS,
                            f"{len(blob['vocab'])} tokens, vocabulary "
                            f"{_short(binding['vocabulary'])}, segments "
                            f"{_short(binding['numeric_edges'])}"))
    problems, notes = [], []
    if arm.config is not None:
        from src.train.run_tokenization_ablation import check_arm_vocab

        try:
            check_arm_vocab(arm.config, blob)
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            problems.append(f"representation: {str(exc)[:200]}")
    for site, directory in arm.sites.items():
        vocab = directory / "vocab.json"
        shard = directory / GEM_EVENTS
        try:
            if vocab.exists():
                compare_binding(artifact_binding(load_vocab_blob(vocab)), binding,
                                what=f"site {site} vocab.json")
            if not shard.exists():
                problems.append(f"site {site}: no {GEM_EVENTS}")
                continue
            recorded = _shard_bindings(shard)
            for hashes in recorded:
                compare_binding(hashes, binding, what=f"site {site} shard rows")
            notes.append(f"{site}: {len(recorded)} binding(s) in the shard match")
            if arm.config is not None and arm.config.get("soft_discretization"):
                width = _soft_width(shard)
                if width < 2:
                    problems.append(f"site {site}: soft arm but the shard has no soft bins")
                else:
                    notes.append(f"soft width {width}")
        except Exception as exc:  # noqa: BLE001
            problems.append(str(exc).split(";")[0][:200])
    checks.append(Check(name, FAIL if problems else PASS,
                        "; ".join(problems) if problems else "; ".join(notes)))
    return checks


def check_value_stats(arm: ArmData, sample: Path | None = None) -> Check:
    """Bound to the arm's vocabulary and segments, fit on train, and covering every
    numeric token of the sample's train stays."""
    import polars as pl

    from src.data.segments import artifact_binding, load_vocab_blob
    from src.data.value_stats import load_value_stats

    name = f"data: {arm.name} value stats"
    if not arm.value_stats.exists():
        return Check(name, FAIL, f"missing {_rel(arm.value_stats)}: run "
                     "`python -m src.data.value_stats --events <gem_events.parquet>`")
    try:
        binding = artifact_binding(load_vocab_blob(arm.vocab))
        stats = load_value_stats(arm.value_stats, expected_vocab_hash=binding["vocabulary"],
                                 expected_segments_hash=binding["numeric_edges"],
                                 expected_fit_partition="train")
    except Exception as exc:  # noqa: BLE001
        return Check(name, FAIL, str(exc)[:200])
    detail = f"{len(stats)} tokens, bound to the vocabulary and segments, fit on train"
    if sample is None:
        return Check(name, PASS, detail)
    numeric = (pl.scan_parquet(sample).filter(pl.col("partition") == "train")
               .select("token", "value").explode(["token", "value"], empty_as_null=True)
               .filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
               .select(pl.col("token").unique()).collect()["token"].to_list())
    missing = sorted({int(t) for t in numeric} - set(stats))
    if missing:
        return Check(name, FAIL, f"{len(missing)} of {len(numeric)} numeric tokens in the "
                     "sample's train stays have no stats (fit on the 24-hour shard?): refit "
                     "on gem_events.parquet")
    return Check(name, PASS, detail + f"; covers all {len(numeric)} numeric tokens of the "
                 "sample")


def textcode_cache_check(arms: Sequence[ArmData]) -> Check | None:
    model = next((a.config.get("textcode_encoder") for a in arms
                  if a.config and a.config.get("tokenizer") == "textcode"), None)
    if model is None:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model, "config.json")
    except Exception:  # noqa: BLE001
        cached = None
    if isinstance(cached, str):
        return Check("data: textcode encoder", PASS, f"{model} is in the local cache")
    return Check("data: textcode encoder", WARN, f"{model} is not in the local Hugging Face "
                 "cache: each textcode launch downloads it (pre-fetch it, see the runbook)")


# ------------------------------------------------------------------ thresholds

def threshold_checks(vocabs: Mapping[str, Mapping], *,
                     thresholds_path: str | Path = THRESHOLDS_PATH) -> list[Check]:
    from src.data.threshold_grid import load_thresholds
    from src.train.run_matrix import MatrixError, edge_distance_table

    try:
        registry = load_thresholds(thresholds_path)
    except Exception as exc:  # noqa: BLE001
        return [Check("thresholds: registry", FAIL, str(exc)[:200])]
    checks = [Check("thresholds: registry", PASS,
                    f"{len(registry['decision'])} decision, {len(registry['control'])} "
                    f"control, {len(registry['competing_risk_cause'])} competing-risk causes")]
    if not vocabs:
        checks.append(Check("thresholds: edge check", FAIL, "no arm vocabulary to check"))
    else:
        try:
            rows = edge_distance_table(vocabs, thresholds=registry)
            controls = sum(r["kind"] == "control" for r in rows)
            checks.append(Check("thresholds: edge check", PASS,
                                f"{controls} control rows off-edge in every arm "
                                f"({', '.join(sorted(vocabs))})"))
        except MatrixError as exc:
            refused = str(exc).split(": a control threshold")[0]
            checks.append(Check("thresholds: edge check", FAIL,
                                f"REFUSED {refused}: decide before launch (remove the "
                                "control from configs/thresholds.yaml or change the arm)"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("thresholds: edge check", FAIL, str(exc)[:200]))
    proposed = [f"{t.concept} {t.rule if t.rule is not None else format(t.value, 'g')}"
                for t in registry["competing_risk_cause"] if t.status == "proposed"]
    checks.append(Check("thresholds: competing-risk causes", WARN if proposed else PASS,
                        f"{len(proposed)} still status: proposed ({', '.join(proposed)}): "
                        "physician confirmation before the first run" if proposed
                        else "all confirmed"))
    return checks


# ------------------------------------------------------------------ schedule

def updates_per_budget(window_lengths: Sequence[int], tcfg: Mapping,
                       budgets: Mapping[str, float], *, ranks: int = EXPECTED_GPUS
                       ) -> tuple[int, dict[str, int]]:
    """(batches per rank per pass, {budget: optimizer updates}) exactly as the launch
    computes them: `DistributedTokenBudgetBatchSampler` over the train windows, then
    `engine.resolve_total_steps` with each budget's passes."""
    import copy

    from src.data.dataset import DistributedTokenBudgetBatchSampler
    from src.train.engine import resolve_total_steps

    sampler = DistributedTokenBudgetBatchSampler(
        list(window_lengths),
        max_batch_tokens=int(tcfg["runtime"].get("token_budget", 0) or 0) or None,
        max_batch_size=tcfg["batch"]["per_gpu"], num_replicas=ranks, rank=0, seed=1)
    batches = len(sampler)
    out = {}
    for budget, passes in budgets.items():
        cfg = copy.deepcopy(dict(tcfg))
        cfg["schedule"] = {**cfg["schedule"], "passes": passes}
        out[budget] = resolve_total_steps(cfg, batches)
    return batches, out


def schedule_check(arm: ArmData, tcfg: Mapping, budgets: Mapping[str, float], *,
                   ranks: int = EXPECTED_GPUS) -> Check:
    """Updates per matrix budget, then `engine.resolve_schedule` per budget (the rules
    `run_tokenization_ablation` refuses a launch by): FAIL when a run's warm-up is as long
    as the run or the run would write no checkpoint at all; WARN when it writes only the
    final one (`runtime.ckpt_every` longer than the run: a crash loses everything) or has
    no warm-up (a one-update run under `schedule.warmup_frac`)."""
    import polars as pl

    from src.train.engine import ScheduleError, resolve_schedule

    lengths: list[int] = []
    for directory in arm.sites.values():
        lengths += (pl.scan_parquet(directory / GEM_EVENTS)
                    .filter(pl.col("partition") == "train")
                    .select(pl.col("token").list.len()).collect().to_series().to_list())
    name = f"schedule: updates per budget [{arm.name}]"
    if not lengths:
        return Check(name, FAIL, "no train windows")
    batches, updates = updates_per_budget(lengths, tcfg, budgets, ranks=ranks)
    accum = int(tcfg["batch"].get("grad_accum", 1))
    parts, refused, no_warmup, final_only = [], [], [], []
    for budget, n in updates.items():
        try:
            plan = resolve_schedule(dict(tcfg), n)
        except ScheduleError as exc:
            refused.append(f"{budget}: {exc}")
            parts.append(f"{budget} {budgets[budget]:g} pass = {n:,} updates (REFUSED)")
            continue
        if plan.warmup_steps == 0 and plan.warmup_source == "warmup_frac":
            no_warmup.append(budget)
        if plan.periodic_checkpoints == 0:
            final_only.append(budget)
        parts.append(f"{budget} {budgets[budget]:g} pass = {n:,} updates, warm-up "
                     f"{plan.warmup_steps} ({plan.warmup_source}), checkpoint every "
                     f"{plan.ckpt_every} ({plan.periodic_checkpoints} periodic"
                     + (" + final)" if plan.final_checkpoint else ", no final)"))
    detail = (f"{len(lengths):,} train windows -> {batches:,} batches per rank per pass "
              f"(per_gpu {tcfg['batch']['per_gpu']}, grad_accum {accum}, {ranks} ranks): "
              + "; ".join(parts))
    if refused:
        return Check(name, FAIL, detail + ". Refused: " + " | ".join(refused))
    if final_only:
        return Check(name, WARN, detail + f": the {', '.join(final_only)} run(s) write no "
                     "periodic checkpoint (runtime.ckpt_every is longer than the run), only "
                     "the final one, so a crash before the end loses the whole run; set "
                     "runtime.ckpt_every to null (runtime.checkpoints_per_run)")
    if no_warmup:
        return Check(name, WARN, detail + f": the {', '.join(no_warmup)} run(s) are too "
                     "short for schedule.warmup_frac to give one warm-up update (one update "
                     "at the peak LR); consider a smaller grad_accum")
    return Check(name, PASS, detail)


# ------------------------------------------------------------------ memory

def target_cache_bytes_per_event() -> int:
    """Bytes per event of the per-rank target cache `ModelDataset._gem_sample` keeps for
    stays split over several windows (cleared each epoch)."""
    import numpy as np

    from src.data.dataset import _TARGET_DTYPES

    return sum(np.dtype(d).itemsize for d in _TARGET_DTYPES.values())


def shard_totals(shard: str | Path) -> dict:
    """Events, windows and stays of the train and validation partitions, and the events
    of stays split over several windows (aggregate)."""
    import polars as pl

    lf = (pl.scan_parquet(shard).filter(pl.col("partition").is_in(["train", "validation"]))
          .select("hosp_id", "partition", "n_windows", pl.col("token").list.len().alias("n")))
    agg = lf.group_by("partition").agg(
        pl.col("n").sum().alias("events"), pl.len().alias("windows"),
        pl.col("hosp_id").n_unique().alias("stays"),
        pl.col("n").filter(pl.col("n_windows") > 1).sum().alias("multi")).collect()
    out = {"train": {"events": 0, "windows": 0, "stays": 0},
           "validation": {"events": 0, "windows": 0, "stays": 0}, "multi_window_events": 0}
    for row in agg.iter_rows(named=True):
        out[row["partition"]] = {k: int(row[k] or 0) for k in ("events", "windows", "stays")}
        out["multi_window_events"] += int(row["multi"] or 0)
    out["events"] = out["train"]["events"] + out["validation"]["events"]
    return out


def write_sample_shard(shard: str | Path, out_dir: str | Path, n_stays: int,
                       policy: Mapping) -> Path:
    """The first `n_stays` train stays and `n_stays // 4` validation stays (sorted keys,
    every window) copied under the governed scratch directory."""
    import polars as pl

    from src.data.cohort import validate_artifact_destination

    out = Path(out_dir) / GEM_EVENTS
    validate_artifact_destination(out, "patient_level_phi", policy)
    lf = pl.scan_parquet(shard)
    keys: list[str] = []
    for partition, n in (("train", n_stays), ("validation", max(1, n_stays // 4))):
        keys += (lf.filter(pl.col("partition") == partition).select("hosp_id").unique()
                 .sort("hosp_id").head(n).collect()["hosp_id"].to_list())
    out.parent.mkdir(parents=True, exist_ok=True)
    lf.filter(pl.col("hosp_id").is_in(keys)).collect().write_parquet(out)
    return out


def resident_bytes() -> int:
    from src.train.pretrain import resident_set_bytes

    return resident_set_bytes()


def peak_resident_bytes() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _build_cache_child(sample: str, queue) -> None:
    """Build the sample's cache in a fresh process, so its peak resident memory is the
    build's and not the parent's history (`mp` spawn target)."""
    try:
        from src.data.dataset import GemCorpus

        gc.collect()
        before = resident_bytes()
        for partition in ("train", "validation"):
            GemCorpus.from_parquet(sample, site="s", partition=partition, cache=True)
        queue.put(("ok", max(peak_resident_bytes() - before, 0)))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {str(exc)[:200]}"))


def measure_sample(sample: Path) -> dict:
    """Build the GemCorpus cache of the sample (train and validation) in a child process
    (its peak resident memory over its pre-build baseline bounds the build cost), then
    load it here as a rank that finds it built would. Returns events, mapped
    (page-cache) bytes, the private resident bytes of the loaded index and the build
    peak."""
    import multiprocessing

    from src.data.dataset import GemCorpus

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_build_cache_child, args=(str(sample), queue))
    child.start()
    status, value = queue.get(timeout=1800)
    child.join(60)
    if status != "ok":
        raise PreflightError(f"cache build failed on the sample: {value}")
    build_peak = int(value)
    gc.collect()
    before = resident_bytes()
    loaded = [GemCorpus.from_parquet(sample, site="s", partition=p, cache=True)
              for p in ("train", "validation")]
    gc.collect()
    private = max(resident_bytes() - before, 0)
    events = sum(c.events for c in loaded)
    mapped = sum(c.nbytes for c in loaded)
    stays = sum(len(c.stays) for c in loaded)
    windows = sum(len(c) for c in loaded)
    del loaded
    return {"events": events, "windows": windows, "stays": stays, "mapped_bytes": mapped,
            "private_bytes": private, "build_peak_bytes": build_peak}


def project_memory(sample: Mapping, totals: Sequence[Mapping], *, ranks: int = EXPECTED_GPUS,
                   target_bytes_per_event: int | None = None,
                   caches_built: bool = False) -> dict:
    """Per-rank and per-node memory for the full corpus from the sample's per-event rates.

    - mapped (page cache, ONE copy per node): mapped bytes per event x events;
    - private per rank: the loaded index's resident bytes per event x events, plus the
      target cache (`target_bytes_per_event` x events of multi-window stays, worst case
      every rank caching every such stay);
    - build transient (rank 0, once, when the cache is not built yet): the build peak per
      event x the largest partition's events.
    `totals` are `shard_totals` per site."""
    if target_bytes_per_event is None:
        target_bytes_per_event = target_cache_bytes_per_event()
    events = max(int(sample["events"]), 1)
    mapped_rate = sample["mapped_bytes"] / events
    private_rate = sample["private_bytes"] / events
    build_rate = sample["build_peak_bytes"] / events
    total_events = sum(t["events"] for t in totals)
    multi = sum(t["multi_window_events"] for t in totals)
    largest = max([max(t["train"]["events"], t["validation"]["events"]) for t in totals]
                  or [0])
    mapped = mapped_rate * total_events
    per_rank = private_rate * total_events + target_bytes_per_event * multi
    build = 0.0 if caches_built else build_rate * largest
    steady = mapped + ranks * per_rank
    return {"events": total_events, "multi_window_events": multi,
            "mapped_bytes_per_event": mapped_rate, "private_bytes_per_event": private_rate,
            "build_bytes_per_event": build_rate,
            "target_bytes_per_event": target_bytes_per_event,
            "mapped_bytes": mapped, "per_rank_bytes": per_rank, "build_bytes": build,
            "node_steady_bytes": steady, "node_peak_bytes": steady + build}


def memory_status(peak_bytes: float, available_bytes: float) -> tuple[str, float]:
    share = peak_bytes / max(available_bytes, 1)
    if share > MEMORY_FAIL:
        return FAIL, share
    if share > MEMORY_WARN:
        return WARN, share
    return PASS, share


def available_memory_bytes() -> tuple[int, int]:
    """(available, total) bytes of RAM: Linux MemAvailable; macOS free + inactive +
    speculative pages (vm_stat)."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) * 1024
        return fields.get("MemAvailable", fields.get("MemFree", 0)), fields.get("MemTotal", 0)
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if sys.platform == "darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             check=False).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in
                 (line.split(":", 1) for line in out.splitlines()[1:] if ":" in line)}
        free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive",
                                              "Pages speculative"))
        return free * page, total
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"), total


def _unescape_mount(path: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), path)


def filesystem_type(path: str | Path) -> tuple[str, bool]:
    """(filesystem type, is local) of the mount holding `path`. Linux: the longest mount
    point prefix in /proc/self/mounts; macOS: `mount`'s ``(type, local, ...)`` flags."""
    target = os.path.realpath(path)
    mounts = Path("/proc/self/mounts")
    if mounts.exists():
        best, fstype = "", "unknown"
        for line in mounts.read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            point = _unescape_mount(parts[1])
            if (target == point or target.startswith(point.rstrip("/") + "/")) \
                    and len(point) >= len(best):
                best, fstype = point, parts[2]
        return fstype, fstype not in NETWORK_FS and not fstype.startswith("fuse")
    out = subprocess.run(["mount"], capture_output=True, text=True, check=False).stdout
    best, fstype, local = "", "unknown", False
    for line in out.splitlines():
        match = re.match(r".+? on (.+) \(([^)]*)\)$", line)
        if not match:
            continue
        point, flags = match.group(1), [f.strip() for f in match.group(2).split(",")]
        if (target == point or target.startswith(point.rstrip("/") + "/")) \
                and len(point) >= len(best):
            best, fstype, local = point, flags[0], "local" in flags
    return fstype, local and fstype not in NETWORK_FS


def check_local_fs(path: Path, what: str) -> Check:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    fstype, local = filesystem_type(probe)
    return Check(f"fs: {what} local filesystem", PASS if local else FAIL,
                 f"{fstype}" + ("" if local else ": the GemCorpus cache needs a local "
                                "filesystem (fcntl lock + memory maps); move the shards"))


def memory_checks(arm: ArmData, *, n_stays: int, scratch: Path, policy: Mapping,
                  ranks: int = EXPECTED_GPUS, keep_scratch: bool = False
                  ) -> tuple[list[Check], dict, dict]:
    """Sample, measure, project (module docstring). Returns the checks, the projection
    and the bytes of cache still to build per shard directory."""
    from src.data.dataset import GEM_CACHE_DIR

    checks = []
    totals, to_build = [], {}
    for site, directory in arm.sites.items():
        shard = directory / GEM_EVENTS
        checks.append(check_local_fs(directory / GEM_CACHE_DIR, f"{site} cache"))
        if not shard.exists():
            checks.append(Check(f"memory: {site}", FAIL, f"no {GEM_EVENTS}"))
            return checks, {}, {}
        totals.append(shard_totals(shard))
    first = next(iter(arm.sites.values())) / GEM_EVENTS
    work = scratch / f"{arm.name}-{os.getpid()}"
    try:
        sample = write_sample_shard(first, work, n_stays, policy)
        measured = measure_sample(sample)
        sample_value_check = check_value_stats(arm, sample)
    finally:
        if not keep_scratch:
            shutil.rmtree(work, ignore_errors=True)
            with contextlib.suppress(OSError):
                scratch.rmdir()                 # only when nothing else is in it
    built = all(any((d / GEM_CACHE_DIR).glob("train-*")) for d in arm.sites.values())
    projection = project_memory(measured, totals, ranks=ranks, caches_built=built)
    for (site, directory), total in zip(arm.sites.items(), totals):
        if not any((directory / GEM_CACHE_DIR).glob("train-*")):
            to_build[directory] = projection["mapped_bytes_per_event"] * total["events"]
    available, total_ram = available_memory_bytes()
    status, share = memory_status(projection["node_peak_bytes"], available)
    checks.append(Check(
        "memory: sample measurement", PASS,
        f"{measured['stays']} stays, {measured['events']:,} events: "
        f"{projection['mapped_bytes_per_event']:.1f} B/event mapped, "
        f"{projection['private_bytes_per_event']:.1f} B/event private per rank, "
        f"build peak <= {projection['build_bytes_per_event']:.1f} B/event"))
    checks.append(Check(
        "memory: projection (train+validation)", status,
        f"{projection['events']:,} events over {len(totals)} site(s): page cache "
        f"{_gib(projection['mapped_bytes'])} once per node, "
        f"{_gib(projection['per_rank_bytes'])} per rank (incl. target cache "
        f"{projection['target_bytes_per_event']} B x {projection['multi_window_events']:,} "
        f"multi-window events), cache build {_gib(projection['build_bytes'])}; node peak "
        f"{_gib(projection['node_peak_bytes'])} = {share:.0%} of {_gib(available)} available "
        f"({_gib(total_ram)} total; warn > {MEMORY_WARN:.0%}, fail > {MEMORY_FAIL:.0%})"))
    checks.append(sample_value_check)
    return checks, projection, to_build


def disk_checks(to_build: Mapping[Path, float], run_root: Path, *, min_free_gb: float
                ) -> list[Check]:
    checks = []
    margin = min_free_gb * 2**30
    for directory, need in to_build.items():
        free = shutil.disk_usage(_existing(directory)).free
        status = PASS if free >= need + margin else FAIL
        checks.append(Check(f"disk: cache {_rel(directory)}", status,
                            f"{_gib(free)} free; cache to build {_gib(need)} + margin "
                            f"{min_free_gb:g} GiB"))
    free = shutil.disk_usage(_existing(run_root)).free
    checks.append(Check(f"disk: checkpoints {_rel(run_root)}",
                        PASS if free >= margin else FAIL,
                        f"{_gib(free)} free; need at least {min_free_gb:g} GiB"))
    return checks


def _existing(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


# ------------------------------------------------------------------ synthetic site

SYNTHETIC_SITE_NAME = "synthetic"


def build_synthetic_site(work: str | Path) -> dict:
    """A synthetic CLIF site tokenized through the real pipeline: episode artifact, 24 h
    vocabulary build, full-hospitalization shard with the same frozen vocabulary and
    value stats fit on its train stays. Returns the paths and the artifact policy."""
    import copy

    import polars as pl

    from src.data.tokenize import tokenize_site
    from src.data.value_stats import compute_value_stats_from_events, write_value_stats
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
        SYNTHETIC_SITE,
    )
    from src.eval.synthetic_bundle import build_synthetic_site as build_tables

    work = Path(work).resolve()
    raw = work / "raw"
    episodes_raw = build_tables(raw)
    hosp = pl.read_parquet(raw / "clif_hospitalization.parquet").with_columns(
        pl.lit("ed").alias("admission_type_category"))
    hosp.write_parquet(raw / "clif_hospitalization.parquet")
    policy = copy.deepcopy(FIXTURE_POLICY)
    for rule in policy["classes"].values():
        rule["directory"] = str(work / rule["directory"])
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(policy))
    data_cfg = yaml.safe_load(DATA_CONFIG_PATH.read_text())
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str(work / "cohort.yaml")
    cfg["artifact_policy"] = str(work / "artifact_policy.yaml")
    cfg["tables"]["adt"] = copy.deepcopy(data_cfg["tables"]["adt"])
    cfg["gem"] = copy.deepcopy(data_cfg["gem"])
    phi = work / "output/intermediate_phi"
    episodes_path = phi / "episodes.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episodes_raw, episodes_path)
    episodes = pl.read_parquet(episodes_path)
    shard_dir = phi / SYNTHETIC_SITE_NAME
    kw = {"episodes": episodes, "artifact_policy": policy, "report": False}
    tokenize_site(cfg, SYNTHETIC_SITE, raw, shard_dir, None, **kw)
    blob = json.loads((shard_dir / "vocab.json").read_text())
    tokenize_site(cfg, SYNTHETIC_SITE, raw, shard_dir, blob, trajectory="hospitalization",
                  **kw)
    stats_path = shard_dir / "gem_value_stats.json"
    write_value_stats(compute_value_stats_from_events(shard_dir / GEM_EVENTS), stats_path,
                      vocab=blob["vocab"], segments=blob["segments"],
                      fit_partition_name="train")
    return {"work": work, "shard_dir": shard_dir, "episodes": episodes_path,
            "value_stats": stats_path, "policy": policy,
            "split_freeze": work / "output/final_no_phi/split_freeze.json"}


# ------------------------------------------------------------------ context length

PROMPT_OVER_CONTEXT_WARN = 0.10


def context_checks(arm: ArmData, *, episodes: Mapping[str, str | Path],
                   cohorts: Mapping[str, str | Path], max_tokens: int = 8192) -> list[Check]:
    """Informational, per site: share of candidate anchors and of extubation time-zero
    prompts whose history exceeds 4,096 / 8,192 / 16,384 tokens (aggregate only,
    `src/eval/context_length.py`). Warns when more than 10% of the prompts exceed
    `max_tokens`: those prompts are read over their last `max_tokens` tokens."""
    from src.eval.context_length import prompt_context_report, shard_context_report

    checks = []
    for site, directory in arm.sites.items():
        shard = directory / GEM_EVENTS
        name = f"context: {arm.name}/{site}"
        if not shard.exists():
            checks.append(Check(name, SKIP, f"no {GEM_EVENTS}"))
            continue
        try:
            anchors = shard_context_report(shard)
            detail = f"candidate anchors over {{4K, 8K, 16K}}: {anchors['candidate_anchors_share_over']}"
            status = PASS
            if site in cohorts and site in episodes:
                prompts = prompt_context_report(shard, cohorts[site], episodes[site])
                share = prompts["share_over"].get(str(max_tokens))
                detail += f"; extubation prompts (n={prompts['n']}): {prompts['share_over']}"
                if isinstance(share, float) and share > PROMPT_OVER_CONTEXT_WARN:
                    status = WARN
                    detail += (f": more than {PROMPT_OVER_CONTEXT_WARN:.0%} of prompts exceed "
                               f"{max_tokens} tokens")
            checks.append(Check(name, status, detail))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(name, WARN, f"{type(exc).__name__}: {str(exc)[:200]}"))
    return checks


# ------------------------------------------------------------------ main

def run_preflight(*, arms: Sequence[ArmData], episodes: Mapping[str, Path],
                  split_freeze: Path | None, skip_gpu: bool, allow_dirty: bool,
                  policy: Mapping, scratch: Path, run_root: Path, memory_arm: str | None,
                  sample_stays: int, min_free_gb: float, smoke_timeout: float,
                  train_config: Path = TRAIN_CONFIG_PATH, keep_scratch: bool = False,
                  thresholds_path: Path = THRESHOLDS_PATH,
                  budgets: Mapping[str, float] | None = None,
                  extubation_cohorts: Mapping[str, Path] | None = None) -> Report:
    import torch

    report = Report()
    report.extend(environment_checks(allow_dirty=allow_dirty))
    report.extend(gpu_checks(skip_gpu=skip_gpu))
    if skip_gpu:
        report.checks.append(ddp_smoke(cpu=True, timeout=smoke_timeout))
    elif torch.cuda.is_available() and torch.cuda.device_count() >= EXPECTED_GPUS:
        report.checks.append(ddp_smoke(cpu=False, timeout=smoke_timeout))
    else:
        report.add("gpu: two-rank DDP smoke", FAIL, "needs two CUDA devices (pass "
                   "--skip-gpu for a CPU gloo rehearsal off the node)")

    report.checks.append(check_split_freeze(split_freeze, episodes, train_config=train_config))
    seen_shards = set()
    for arm in arms:
        report.extend(check_arm_binding(arm))
        for site, directory in arm.sites.items():
            shard = directory / GEM_EVENTS
            if site in episodes and shard.exists() and shard.resolve() not in seen_shards:
                seen_shards.add(shard.resolve())
                if Path(episodes[site]).exists():
                    check = check_shard_partitions(site, shard, episodes[site])
                    check.name = f"split: {arm.name}/{site} shard partitions"
                    report.checks.append(check)
    textcode = textcode_cache_check(arms)
    if textcode is not None:
        report.checks.append(textcode)

    from src.data.segments import load_vocab_blob

    vocabs = {a.name: load_vocab_blob(a.vocab) for a in arms if a.vocab.exists()}
    report.extend(threshold_checks(vocabs, thresholds_path=thresholds_path))

    target = next((a for a in arms if a.name == memory_arm), arms[0] if arms else None)
    to_build: dict = {}
    if target is None:
        report.add("memory: projection", FAIL, "no arm to measure")
    else:
        try:
            checks, _, to_build = memory_checks(target, n_stays=sample_stays,
                                                scratch=scratch, policy=policy,
                                                keep_scratch=keep_scratch)
            for check in checks:
                if check.name.startswith("memory"):
                    check.name = f"{check.name} [{target.name}]"
            report.extend(checks)
        except Exception as exc:  # noqa: BLE001
            report.add(f"memory: projection [{target.name}]", FAIL,
                       f"{type(exc).__name__}: {str(exc)[:200]}")
    for arm in arms:
        if arm is not target:
            report.checks.append(check_value_stats(arm))
    if target is not None and budgets is None:
        report.add(f"schedule: updates per budget [{target.name}]", SKIP,
                   "no matrix budgets (synthetic site)")
    elif target is not None:
        tcfg = yaml.safe_load(Path(train_config).read_text())
        try:
            report.checks.append(schedule_check(target, tcfg, budgets))
        except Exception as exc:  # noqa: BLE001
            report.add(f"schedule: updates per budget [{target.name}]", FAIL,
                       f"{type(exc).__name__}: {str(exc)[:200]}")
    if target is not None:
        report.extend(context_checks(target, episodes=episodes, cohorts=extubation_cohorts or {}))
    report.extend(disk_checks(to_build, run_root, min_free_gb=min_free_gb))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="L40 pre-flight: environment, GPUs, data binding, split freeze, "
                    "thresholds, memory and disk (aggregate output only)")
    ap.add_argument("--episodes", action="append", default=[], metavar="[SITE=]PATH",
                    help="a site's episode artifact (repeat per site; a bare PATH is the "
                         f"first launch site's; default {DEFAULT_EPISODES})")
    ap.add_argument("--split-freeze", default=str(DEFAULT_SPLIT_FREEZE),
                    help="the KTD6 split freeze record (read, or written with "
                         "--write-split-freeze)")
    ap.add_argument("--write-split-freeze", action="store_true",
                    help="compute the split freeze from --episodes (and --audit-blind) "
                         "and write it, then exit")
    ap.add_argument("--audit-blind", default=None,
                    help="the U13 blind-stage export JSON: its held-out arm counts are "
                         "recorded in the freeze")
    ap.add_argument("--approver", default=None, help="who decided the held-out share")
    ap.add_argument("--date", default=None, help="decision date (default: today, UTC)")
    ap.add_argument("--force", action="store_true",
                    help="with --write-split-freeze: replace a freeze of a different split")
    ap.add_argument("--matrix", default=str(MATRIX_PATH))
    ap.add_argument("--ablation-config", default=str(ABLATION_PATH))
    ap.add_argument("--train-config", default=str(TRAIN_CONFIG_PATH))
    ap.add_argument("--thresholds", default=str(THRESHOLDS_PATH))
    ap.add_argument("--arm", action="append", default=[], metavar="ARM",
                    help="check only these tokenization arms (default: every arm_data arm)")
    ap.add_argument("--arm-dir", action="append", default=[], metavar="ARM[:SITE]=DIR",
                    help="override an arm's shard directory (default: the matrix arm_data)")
    ap.add_argument("--memory-arm", default="clinical_soft",
                    help="the arm whose shards are sampled for the memory projection")
    ap.add_argument("--sample-stays", type=int, default=500,
                    help="train stays in the memory sample (plus a quarter as many "
                         "validation stays)")
    ap.add_argument("--scratch", default=str(DEFAULT_SCRATCH),
                    help="governed scratch directory for the sample copy (deleted after)")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--run-root", default=None,
                    help="checkpoint root for the disk check (default: matrix launch.run_root)")
    ap.add_argument("--min-free-gb", type=float, default=50.0,
                    help="free space required beyond the cache still to build")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="off the node: skip the GPU checks and run the two-rank smoke over "
                         "gloo on CPU")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="warn instead of failing on modified tracked files")
    ap.add_argument("--smoke-timeout", type=float, default=SMOKE_TIMEOUT_S)
    ap.add_argument("--extubation-cohort", action="append", default=[], metavar="[SITE=]PATH",
                    help="a site's extubation cohort artifact: adds the share of time-zero "
                         "prompts over 4K/8K/16K tokens to the context check")
    ap.add_argument("--synthetic", action="store_true",
                    help="build a synthetic site in a temporary directory and check it")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    first_site = matrix["launch"]["sites"][0]
    try:
        if args.synthetic:
            return _main_synthetic(args)
        episodes = _site_paths(args.episodes or [str(DEFAULT_EPISODES)], first_site)
        if args.write_split_freeze:
            record = build_split_freeze(episodes, approver=args.approver, date=args.date,
                                        audit_blind=args.audit_blind,
                                        train_config=args.train_config)
            path = write_split_freeze(args.split_freeze, record, force=args.force)
            _print_freeze(record, path)
            return 0
        arms = matrix_arms(matrix_path=args.matrix, ablation_path=args.ablation_config,
                           overrides=args.arm_dir, only=args.arm or None)
        policy = yaml.safe_load(POLICY_PATH.read_text())
        report = run_preflight(
            arms=arms, episodes=episodes, split_freeze=Path(args.split_freeze),
            skip_gpu=args.skip_gpu, allow_dirty=args.allow_dirty, policy=policy,
            scratch=Path(args.scratch),
            run_root=Path(args.run_root or matrix["launch"]["run_root"]),
            memory_arm=args.memory_arm, sample_stays=args.sample_stays,
            min_free_gb=args.min_free_gb, smoke_timeout=args.smoke_timeout,
            train_config=Path(args.train_config), keep_scratch=args.keep_scratch,
            thresholds_path=Path(args.thresholds), budgets=matrix["budgets"],
            extubation_cohorts=_site_paths(args.extubation_cohort, first_site)
            if args.extubation_cohort else None)
    except PreflightError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(report.table())
    return 1 if report.failed else 0


def _print_freeze(record: Mapping, path: Path) -> None:
    print(f"wrote {_rel(path)}: approver {record['approver']}, date {record['date']}")
    for site, summary in record["sites"].items():
        print(f"  {site}: split {_short(summary['split_sha256'])}, "
              f"{summary['eligible_episodes']} eligible episodes, shares "
              f"{summary['observed_shares']}, configured {summary['configured_proportions']}, "
              f"seed {summary['split_seed']}")
    if "audit_blind" in record:
        print(f"  held-out arms (audit blind stage): {record['audit_blind']['held_out_arms']}")


def _main_synthetic(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="preflight_synthetic_") as td:
        site = build_synthetic_site(td)
        episodes = {SYNTHETIC_SITE_NAME: site["episodes"]}
        record = build_split_freeze(episodes, approver="synthetic rehearsal",
                                    train_config=args.train_config)
        write_split_freeze(site["split_freeze"], record)
        arm = ArmData("synthetic", {SYNTHETIC_SITE_NAME: site["shard_dir"]},
                      site["value_stats"], None)
        report = run_preflight(
            arms=[arm], episodes=episodes, split_freeze=site["split_freeze"],
            skip_gpu=args.skip_gpu, allow_dirty=args.allow_dirty, policy=site["policy"],
            scratch=site["work"] / "output/intermediate_phi/preflight_scratch",
            run_root=site["work"], memory_arm="synthetic",
            sample_stays=args.sample_stays, min_free_gb=min(args.min_free_gb, 1.0),
            smoke_timeout=args.smoke_timeout, train_config=Path(args.train_config),
            thresholds_path=Path(args.thresholds))
        print(report.table())
        return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
