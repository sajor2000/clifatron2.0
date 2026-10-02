"""Tokenization ablation driver — one arm of configs/tokenization_ablation.yaml per run.

Every arm trains through the pretrain path (U6, KTD8): its own events shard and frozen
vocab.json are loaded by `pretrain.build_loaders` (shard rows and value stats bound to
that vocabulary), and the model is `pretrain.Model` — the same masked next-event /
competing-risk / threshold-hazard / value losses, weighted by `configs/model.yaml`
`heads.*.weight` — trained by `engine.train`. Only the INPUT representation varies:

- ``fused``: fused ``concept=bin`` tokens; hard ids, or soft [T, K] bins when the arm sets
  ``soft_discretization`` (clinical segments or deciles, per the arm's ``scheme``).
- ``continuous_fused``: edgeless concept ids + the normalized current-value channel; the
  threshold head's value bins are the primary clinical segments'
  (`src/data/tokenize_continuous.py`).
- ``textcode``: a frozen text embedding of every fused id's generated description +
  a trainable projection (`src/data/tokenize_textcode.py`, `TextCodeEncoder`).

``freeze_trunk`` freezes the transformer blocks only when an init checkpoint (bound to
the arm's vocabulary) is given; otherwise it warns and the trunk stays trainable —
freezing a random trunk would train the heads on noise.

Usage:
    torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm clinical_soft
    # per-arm data paths come from the config; override on the CLI:
    torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation --arm global_deciles \
        --events <dir>/events_with_outcomes.parquet --vocab <dir>/vocab.json \
        --value-stats <dir>/value_stats.json
"""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.segments import (
    artifact_binding,
    compare_binding,
    n_value_bins as vocab_n_value_bins,
    segments_hash,
    vocab_segments,
)
from src.data.tokenize_continuous import (
    REPRESENTATION as CONTINUOUS_FUSED,
    primary_n_value_bins,
    primary_segments,
)
from src.data.tokenize_textcode import DEFAULT_TEXTCODE_ENCODER, TextEncoder, textcode_table
from src.model.encoder import count_params
from src.model.encoder_continuous import ContinuousFusedEncoder
from src.model.encoder_textcode import TextCodeEncoder
from src.train.checkpoint import load_checkpoint
from src.train.engine import TrainConfig, is_distributed, setup_ddp, train
from src.train.pretrain import Loaders, Model, build_loaders, build_scheduler

TOKENIZERS = ("fused", CONTINUOUS_FUSED, "textcode")
SCHEMES = ("clinical_segment", "decile_ablation")   # configs/data.yaml value_binning.scheme
PATH_KEYS = ("events", "vocab", "value_stats")
SHARED_ARM_KEYS = ("freeze_trunk", "init_checkpoint")


class TokenizationAblationModel(Model):
    """`pretrain.Model` (objective, masks, config weights) with the arm's encoder."""

    def __init__(self, vocab_size: int, n_targets: int, mcfg: dict, arm_cfg: dict, *,
                 n_value_bins: int, text_table=None):
        tokenizer = arm_cfg.get("tokenizer", "fused")
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"unknown arm tokenizer {tokenizer!r}; expected one of {TOKENIZERS}")
        if tokenizer == CONTINUOUS_FUSED:
            encoder = ContinuousFusedEncoder(vocab_size, mcfg)
        elif tokenizer == "textcode":
            if text_table is None:
                raise ValueError("the TextCode arm needs its frozen description embedding "
                                 "table (tokenize_textcode.textcode_table)")
            encoder = TextCodeEncoder(vocab_size, mcfg, text_table)
        else:
            encoder = None
        super().__init__(vocab_size, n_targets, mcfg, n_value_bins=n_value_bins,
                         encoder=encoder)
        self.tokenizer = tokenizer


@dataclass
class ArmRun:
    name: str
    arm: dict
    model: TokenizationAblationModel
    loaders: Loaders
    vocab_blob: dict
    binding: dict[str, str]
    n_value_bins: int


def resolve_arm(abl: dict, name: str, *, events=None, vocab=None, value_stats=None,
                init_checkpoint=None, primary_vocab=None) -> dict:
    """The arm's config: shared ``freeze_trunk`` / ``init_checkpoint``, then the arm's own
    keys, then CLI overrides of its data paths. Every arm names its own events shard and
    vocabulary (one ``--data`` for all arms trained every arm on one tokenization)."""
    arms = abl.get("arms") or {}
    if name not in arms:
        raise SystemExit(f"Unknown arm: {name}. Choices: {list(arms)}")
    shared = abl.get("shared") or {}
    arm = {key: shared[key] for key in SHARED_ARM_KEYS if key in shared}
    arm.update(copy.deepcopy(arms[name]))
    arm["name"] = name
    for key, value in (("events", events), ("vocab", vocab), ("value_stats", value_stats),
                       ("init_checkpoint", init_checkpoint), ("primary_vocab", primary_vocab)):
        if value is not None:
            arm[key] = str(value)
    if arm.get("tokenizer") not in TOKENIZERS:
        raise SystemExit(f"arm {name}: tokenizer must be one of {TOKENIZERS}")
    if arm.get("scheme") not in SCHEMES:
        raise SystemExit(f"arm {name}: scheme must be one of {SCHEMES} (configs/data.yaml)")
    for key in ("events", "vocab"):
        if not arm.get(key):
            raise SystemExit(f"arm {name} needs its own {key} path "
                             f"(arms.{name}.{key} or --{key.replace('_', '-')})")
    return arm


def check_arm_vocab(arm: dict, blob: dict) -> int:
    """Refuse a vocabulary that is not the arm's representation/scheme; return the
    threshold head's ``n_value_bins`` (the primary segments' for continuous-fused)."""
    name, scheme = arm["name"], arm["scheme"]
    continuous = blob.get("representation") == CONTINUOUS_FUSED
    if arm["tokenizer"] == CONTINUOUS_FUSED and not continuous:
        raise SystemExit(f"arm {name} needs a continuous-fused vocabulary; derive it from "
                         "the primary shard with `python -m src.data.tokenize_continuous`")
    if continuous and arm["tokenizer"] != CONTINUOUS_FUSED:
        raise SystemExit(f"arm {name} ({arm['tokenizer']}) was given a continuous-fused "
                         "vocabulary")
    if continuous:
        segments = primary_segments(blob)  # verifies the recorded primary_segments hash
        sources = blob.get("primary_binning_sources") or {}
        if arm.get("primary_vocab"):
            primary = json.loads(Path(arm["primary_vocab"]).read_text())
            if segments_hash(vocab_segments(primary)) != segments_hash(segments):
                raise SystemExit(f"arm {name}: its primary segments are not those of "
                                 f"{arm['primary_vocab']}")
        value_bins = primary_n_value_bins(blob)
    else:
        sources = blob.get("binning_sources") or {}
        value_bins = vocab_n_value_bins(blob)
    has_csv = "csv" in sources.values()
    if (scheme == "clinical_segment") != has_csv:
        raise SystemExit(
            f"arm {name}: scheme {scheme} but its vocabulary "
            f"{'uses' if has_csv else 'has no'} physician CSV segments; point it at the "
            f"{scheme} tokenization"
        )
    return value_bins


def apply_freeze_trunk(model: Model, freeze: bool, init_checkpoint) -> bool:
    """Freeze the transformer trunk (blocks + final norm) only when it was initialised
    from a checkpoint; embeddings/projections and heads stay trainable. Without an init
    checkpoint, warn and keep it trainable. Returns whether the trunk was frozen."""
    if not freeze:
        return False
    if not init_checkpoint:
        warnings.warn(
            "freeze_trunk is set but no init checkpoint was given: a randomly initialised "
            "trunk would be frozen, so it is kept trainable",
            UserWarning,
            stacklevel=2,
        )
        return False
    for module in (model.enc.blocks, model.enc.ln_f):
        for param in module.parameters():
            param.requires_grad_(False)
    return True


def setup_arm(arm: dict, *, mcfg: dict, tcfg: dict, n_targets: int, device,
              text_encoder: TextEncoder | None = None, init_checkpoint=None,
              dry_run: bool = False, is_main: bool = True, seed: int = 42) -> ArmRun:
    """Model + loaders for one resolved arm (see `resolve_arm`)."""
    torch.manual_seed(seed)
    blob = json.loads(Path(arm["vocab"]).read_text())
    binding = artifact_binding(blob)  # refuses a pre-v2 vocabulary
    value_bins = check_arm_vocab(arm, blob)
    vocab_size = int(mcfg["trunk"].get("target_vocab", 10000))
    if max(blob["vocab"].values()) >= vocab_size:
        raise SystemExit(f"arm {arm['name']}: vocabulary ids exceed the model's "
                         f"target_vocab {vocab_size}")
    text_table = None
    if arm["tokenizer"] == "textcode":
        text_table = textcode_table(
            blob, vocab_size, encode=text_encoder,
            model_name=arm.get("textcode_encoder", DEFAULT_TEXTCODE_ENCODER))
    model = TokenizationAblationModel(vocab_size, n_targets, mcfg, arm,
                                      n_value_bins=value_bins, text_table=text_table)
    init = init_checkpoint if init_checkpoint is not None else arm.get("init_checkpoint")
    if init:
        loaded = load_checkpoint(init, "cpu")
        compare_binding(loaded.get("vocab_binding"), binding, what="init checkpoint")
        model.load_state_dict(loaded["model"])
    apply_freeze_trunk(model, bool(arm.get("freeze_trunk", False)), init)
    model.to(device)
    loaders = build_loaders(
        arm["events"],
        binding=binding,
        tcfg=tcfg,
        mcfg=mcfg,
        vocab_size=vocab_size,
        value_stats_path=arm.get("value_stats"),
        dry_run=dry_run,
        is_main=is_main,
        soft=bool(arm.get("soft_discretization", False)),
        value_channel=arm["tokenizer"] == CONTINUOUS_FUSED,
    )
    return ArmRun(arm["name"], arm, model, loaders, blob, binding, value_bins)


def train_arm(run: ArmRun, *, tcfg: dict, mcfg: dict, device, total_steps=None, lr=None,
              out_dir=None, local: int = 0):
    """`engine.train` over the arm's loaders: AdamW on the trainable parameters, the
    pretrain warmup+cosine schedule, checkpoints bound to the arm's vocabulary."""
    total_steps = int(total_steps if total_steps is not None else run.arm["total_steps"])
    lr = float(lr if lr is not None else run.arm["lr"])
    tcfg = copy.deepcopy(tcfg)
    if out_dir is not None:
        tcfg["runtime"]["ckpt_dir"] = str(Path(out_dir) / "checkpoints")
    model = run.model
    if tcfg["runtime"].get("compile", mcfg.get("compile", False)) and torch.cuda.is_available():
        model = torch.compile(model, dynamic=True)
    if is_distributed():
        model = DDP(model, device_ids=[local])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )
    warmup = min(int(tcfg["schedule"].get("warmup_steps", 2000)), total_steps)
    scheduler = build_scheduler(opt, total_steps, warmup)
    train_cfg = TrainConfig({}, tcfg, mcfg, total_steps)
    return train(model, run.loaders.train, run.loaders.validation, opt, scheduler,
                 train_cfg, device, seed=42, vocab_binding=run.binding)


def masked_target_counts(batch: dict) -> dict[str, int]:
    """Aggregate supervised-position counts of a prepared batch (`engine._prepare_batch`)."""
    def count(key: str) -> int:
        return int(batch[key].sum()) if key in batch else 0

    return {"ntp": count("ntp_mask"), "value": count("val_mask"),
            "cr": count("cr_mask"), "th": count("th_mask")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--events", default=None, help="override the arm's events shard")
    ap.add_argument("--vocab", default=None, help="override the arm's tokenizer-v2 vocab.json")
    ap.add_argument("--value-stats", default=None, help="override the arm's value_stats.json")
    ap.add_argument("--primary-vocab", default=None,
                    help="continuous_fused: the primary clinical vocab.json its segments "
                         "must match")
    ap.add_argument("--init-checkpoint", default=None,
                    help="initialise from a checkpoint bound to the arm's vocabulary "
                         "(required for freeze_trunk to freeze anything)")
    ap.add_argument("--ablation-config", default="configs/tokenization_ablation.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--out", default="results/tokenization_ablation")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the arm's model and loaders, print shapes, and exit")
    args = ap.parse_args()

    local, is_main = setup_ddp()
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{local}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    tcfg = yaml.safe_load(Path(args.train_config).read_text())
    n_targets = len(yaml.safe_load(Path(args.data_config).read_text())["target_concepts"])
    arm = resolve_arm(abl, args.arm, events=args.events, vocab=args.vocab,
                      value_stats=args.value_stats, init_checkpoint=args.init_checkpoint,
                      primary_vocab=args.primary_vocab)
    run = setup_arm(arm, mcfg=mcfg, tcfg=tcfg, n_targets=n_targets, device=dev,
                    dry_run=args.dry_run, is_main=is_main)

    if is_main:
        total = count_params(run.model)
        trainable = sum(p.numel() for p in run.model.parameters() if p.requires_grad)
        print(f"[{args.arm}] {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable")
        print(f"  tokenizer: {arm['tokenizer']}  scheme: {arm['scheme']}  "
              f"soft: {bool(arm.get('soft_discretization'))}  "
              f"n_value_bins: {run.n_value_bins}")
        print(f"  description: {arm.get('description', '')}")
    if args.dry_run:
        if is_main:
            dataset = run.loaders.train_dataset
            batch = run.loaders.train.collate_fn([dataset[0], dataset[min(1, len(dataset) - 1)]])
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {list(value.shape)}")
        if is_distributed():
            dist.destroy_process_group()
        return

    out_dir = Path(args.out) / args.arm
    _, manifest = train_arm(run, tcfg=tcfg, mcfg=mcfg, device=dev,
                            total_steps=args.total_steps, out_dir=out_dir, local=local)
    if is_main:
        # Aggregate-only run record (no row-level data).
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run.json").write_text(json.dumps({
            "arm": args.arm, "run_id": manifest.run_id, "ledger": manifest.ledger,
            "validation": manifest.validation, "vocab_binding": run.binding,
            "n_value_bins": run.n_value_bins,
        }, indent=2))
        print(f"[{args.arm}] training complete. Run ID: {manifest.run_id}")
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
