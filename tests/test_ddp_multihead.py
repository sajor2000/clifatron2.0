"""U2: DDP-safe multi-head objective and gradient accumulation (R31, KTD2).

Every distributed test here launches two real processes on CPU over the gloo backend
(the L40 run is the same code over nccl) and drives the production entry points:
`engine.setup_ddp` / `engine.wrap_ddp` / `engine.train` with `pretrain.Model`, and
`run_tokenization_ablation.train_arm` for the ablation arms.

What is proven:
  - a head with zero weight, or a rank whose batch has no supervised anchor, still
    completes optimizer updates (the skipped head touches its parameters with a
    zero-valued term, so every rank marks the same parameters ready);
  - accumulation runs non-boundary microsteps under `no_sync` and yields the gradient a
    single process computes over the same samples;
  - an epoch whose microbatch count is not a multiple of the accumulation factor, and a
    stop at the update limit, leave both ranks with identical parameters;
  - U4 (KTD5): under the next-token warm-up of the curriculum both ranks apply the same
    weights on the same update, complete their updates, and leave every head bit-identical
    to its initialisation until the transition starts.

All data is synthetic. Each launch picks a free port, is killed at a deadline instead of
hanging, and never leaves a child process behind.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
VOCAB, N_TARGETS, N_VALUE_BINS, N_TIME_BINS = 32, 2, 5, 4
BINDING = {"tokenizer_version": "2", "vocabulary": "a" * 64, "numeric_edges": "b" * 64}
LAUNCH_TIMEOUT_S = 180
CPU = torch.device("cpu")


def tiny_mcfg(*, curriculum: str | None = None, **weights: float) -> dict:
    """A one-layer d16 `pretrain.Model` config; `weights` overrides `heads.<name>.weight`,
    `curriculum` sets the model config's `curriculum` (absent: no curriculum)."""
    heads = {
        "next_event": {"enabled": True, "weight": 0.2},
        "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": N_TIME_BINS},
        "threshold_hazard": {"enabled": True, "weight": 1.0, "n_time_bins": N_TIME_BINS,
                             "threshold_embed_dim": 4},
        "value_regression": {"enabled": True, "weight": 0.5},
    }
    for name, weight in weights.items():
        heads[name]["weight"] = weight
    mcfg = {
        "trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0,
                  "rope_base": 10000.0, "tied_embeddings": False},
        "heads": heads,
    }
    if curriculum is not None:
        mcfg["curriculum"] = curriculum
    return mcfg


def tiny_tcfg(ckpt_dir, *, grad_accum: int = 1, ckpt_every: int = 1000, val_every: int = 1000,
              grad_clip: float | None = 1.0) -> dict:
    return {
        "optimizer": {"lr": 1e-2, "weight_decay": 0.1, "betas": [0.9, 0.95],
                      "grad_clip": grad_clip},
        "schedule": {"warmup_steps": 1, "total_steps": 3, "cosine_decay": True},
        "batch": {"per_gpu": 2, "grad_accum": grad_accum},
        "runtime": {"precision": "bf16", "num_workers": 0, "log_every": 1000,
                    "ckpt_every": ckpt_every, "ckpt_dir": str(ckpt_dir)},
        "eval_schedule": {"val_every": val_every},
    }


def synthetic_batch(seed: int, *, supervised: bool = True, n: int = 2, t: int = 6) -> dict:
    """One collated microbatch in the engine's batch contract (`engine._prepare_batch`).

    `supervised=False` is a batch with no supervised anchor: every competing-risk label
    and threshold query is masked out, as for stays without joined outcomes."""
    g = torch.Generator().manual_seed(seed)
    token = torch.randint(1, VOCAB, (n, t), generator=g)
    ntp_mask = torch.ones(n, t, dtype=torch.bool)
    ntp_mask[:, -1] = False
    value_mask = torch.rand(n, t, generator=g) > 0.5
    value_mask[:, 0] = True
    anchor_mask = torch.full((n,), supervised, dtype=torch.bool)
    return {
        "input_ids": token,
        "attention_mask": torch.ones(n, t, dtype=torch.bool),
        "pos_min": torch.arange(t).repeat(n, 1) * 5,
        "anchor_idx": torch.full((n,), t - 1, dtype=torch.long),
        "ntp_target": torch.roll(token, -1, dims=1),
        "ntp_mask": ntp_mask,
        "value_target": torch.randn(n, t, generator=g),
        "value_mask": value_mask,
        # Continuous-fused arm input channel (ignored by the fused / TextCode encoders).
        "input_value": torch.randn(n, t, generator=g),
        "input_value_mask": value_mask.clone(),
        "cr_mask": anchor_mask,
        "cr_type": torch.randint(-1, N_TARGETS + 1, (n,), generator=g),
        "cr_bin": torch.randint(0, N_TIME_BINS, (n,), generator=g),
        "th_mask": anchor_mask.clone(),
        "th_target": torch.randint(0, N_TARGETS, (n,), generator=g),
        "th_tau": torch.randint(0, N_VALUE_BINS, (n,), generator=g),
        "th_dir": torch.randint(0, 2, (n,), generator=g),
        "th_crossed": torch.randint(-1, N_TIME_BINS, (n,), generator=g),
        "th_observed_bin": torch.randint(1, N_TIME_BINS + 1, (n,), generator=g),
    }


class _Microbatches(Dataset):
    """Pre-collated microbatches; `DataLoader(batch_size=None)` yields them unchanged."""

    def __init__(self, batches: list[dict]):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def microbatch_loader(batches: list[dict], *, distributed: bool) -> DataLoader:
    """`DistributedSampler(shuffle=False)` gives rank r the microbatches r, r+2, ..."""
    dataset = _Microbatches(batches)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=None, sampler=sampler, shuffle=False)


def build_model(mcfg: dict, seed: int = 0):
    from src.train.pretrain import Model

    torch.manual_seed(seed)
    return Model(VOCAB, N_TARGETS, mcfg, n_value_bins=N_VALUE_BINS)


def record_weights(model) -> list[tuple[float, ...]]:
    """(next-event, competing-risk, threshold, value) weights a bare `pretrain.Model`
    applied on each training forward."""
    from src.train.curriculum import HEADS

    used: list[tuple[float, ...]] = []
    model.register_forward_hook(
        lambda module, *_: used.append(tuple(module.loss_weights[head] for head in HEADS)))
    return used


HEAD_PREFIXES = ("cr.", "th.", "vr.")


def head_state(state: dict, prefixes: tuple[str, ...] = HEAD_PREFIXES) -> dict:
    """The time-to-event and value heads' entries of a `pretrain.Model` state dict."""
    return {name: value for name, value in state.items() if name.startswith(prefixes)}


def accumulated_gradient(model, loader, *, grad_accum: int, rank: int = 0) -> dict:
    """The gradient `_train_one_epoch` applies in its (single) update: plain SGD at lr 1
    with no clipping moves each parameter by exactly minus that gradient."""
    from src.train.engine import TrainConfig, _train_one_epoch

    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig({}, tiny_tcfg(td, grad_accum=grad_accum, grad_clip=None),
                          {"compile": False}, total_steps=1)
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        _, _, _, _, updates = _train_one_epoch(model, loader, opt, scheduler, 0, cfg, CPU,
                                               rank=rank)
    if updates != 1:
        raise AssertionError(f"expected one accumulated update, got {updates}")
    return {k: before[k] - v.detach() for k, v in model.named_parameters()}


# ------------------------------------------------------------- rank-side scenarios

def _wrap(model, local: int):
    from src.train.engine import wrap_ddp

    return wrap_ddp(model, CPU, local)


def _record_sync(ddp) -> list[bool]:
    """Whether each forward was a synchronized step (False inside `no_sync`)."""
    flags: list[bool] = []
    ddp.module.register_forward_pre_hook(
        lambda *_: flags.append(bool(ddp.require_backward_grad_sync)))
    return flags


def _state(ddp) -> dict:
    return {k: v.detach().clone() for k, v in ddp.module.state_dict().items()}


def _train_pretrain_model(local: int, out_dir: Path, mcfg: dict, batches: list[dict], *,
                          total_steps: int, grad_accum: int = 1,
                          ckpt_every: int = 1000) -> dict:
    """`engine.train` over the DDP-wrapped `pretrain.Model` with the pretrain optimizer."""
    from src.train.engine import TrainConfig, train
    from src.train.pretrain import build_optimizer, build_scheduler

    bare = build_model(mcfg)
    weights = record_weights(bare)
    model = _wrap(bare, local)
    flags = _record_sync(model)
    tcfg = tiny_tcfg(out_dir / "ckpt", grad_accum=grad_accum, ckpt_every=ckpt_every)
    opt = build_optimizer(model, lr=tcfg["optimizer"]["lr"],
                          weight_decay=tcfg["optimizer"]["weight_decay"],
                          betas=tcfg["optimizer"]["betas"])
    scheduler = build_scheduler(opt, total_steps, 1)
    _, manifest = train(model, microbatch_loader(batches, distributed=True), None, opt,
                        scheduler, TrainConfig({}, tcfg, mcfg, total_steps), CPU,
                        vocab_binding=BINDING)
    return {"updates": manifest.ledger["optimizer_updates"], "state": _state(model),
            "sync": flags, "weights": weights}


def scenario_tte_weights_zero(local: int, out_dir: Path) -> dict:
    """Pure next-token recipe: both time-to-event heads carry zero weight on both ranks."""
    mcfg = tiny_mcfg(competing_risk=0.0, threshold_hazard=0.0)
    batches = [synthetic_batch(i) for i in range(6)]          # 3 microbatches per rank
    return _train_pretrain_model(local, out_dir, mcfg, batches, total_steps=3, ckpt_every=3)


def scenario_rank_without_anchor(local: int, out_dir: Path) -> dict:
    """Rank 1 (odd microbatches) never sees a supervised anchor; rank 0 always does."""
    batches = [synthetic_batch(i, supervised=i % 2 == 0) for i in range(6)]
    return _train_pretrain_model(local, out_dir, tiny_mcfg(), batches, total_steps=3)


ACCUM_BATCHES = 8   # 4 microbatches per rank, one accumulated update


def accumulation_batches() -> list[dict]:
    return [synthetic_batch(100 + i, supervised=i % 3 != 0) for i in range(ACCUM_BATCHES)]


def scenario_accumulation_gradient(local: int, out_dir: Path) -> dict:
    model = _wrap(build_model(tiny_mcfg()), local)
    flags = _record_sync(model)
    loader = microbatch_loader(accumulation_batches(), distributed=True)
    grad = accumulated_gradient(model, loader, grad_accum=ACCUM_BATCHES // 2, rank=local)
    return {"grad": {k.removeprefix("module."): v for k, v in grad.items()}, "sync": flags}


def scenario_partial_epoch(local: int, out_dir: Path) -> dict:
    """5 microbatches per rank at accumulation 2: updates at microbatch 2, 4 and the
    partial 5th; the second epoch stops at the update limit after microbatch 4."""
    batches = [synthetic_batch(200 + i, supervised=i % 4 != 1) for i in range(10)]
    return _train_pretrain_model(local, out_dir, tiny_mcfg(), batches, total_steps=5,
                                 grad_accum=2)


CURRICULUM_STEPS = 20      # warm-up: updates 0..2; transition: update 3; configured from 4
CURRICULUM_ACCUM = 2


def scenario_curriculum_warmup(local: int, out_dir: Path) -> dict:
    """The curriculum under DDP at accumulation 2, a checkpoint after every update. Rank 1
    never sees a supervised anchor, so its heads are skipped for the whole run."""
    batches = [synthetic_batch(400 + i, supervised=i % 2 == 0) for i in range(8)]
    return _train_pretrain_model(local, out_dir, tiny_mcfg(curriculum="ntp_then_tte"),
                                 batches, total_steps=CURRICULUM_STEPS,
                                 grad_accum=CURRICULUM_ACCUM, ckpt_every=1)


def adapter_batch(seed: int, *, supervised: bool = True) -> dict:
    """`synthetic_batch` in the per-anchor minutes contract the adapter reads: one anchor
    per row, label times in minutes since the anchor (no pre-binned keys)."""
    batch = synthetic_batch(seed, supervised=supervised)
    for key in ("cr_bin", "th_crossed", "th_observed_bin"):
        del batch[key]
    g = torch.Generator().manual_seed(seed + 10_000)
    n = batch["input_ids"].size(0)
    batch["anchor_batch_idx"] = torch.arange(n)
    batch["cr_time_min"] = torch.randint(0, 48 * 60, (n,), generator=g)
    batch["th_anchor"] = torch.arange(n)
    batch["th_event"] = torch.rand(n, generator=g) > 0.5
    batch["th_time_min"] = torch.randint(0, 48 * 60, (n,), generator=g)
    return batch


def tiny_adapter_model(*, freeze: bool, curriculum: str = "none", seed: int = 0):
    """`run_arm.AdapterModel` on a two-layer GPT2 backbone (the CLIFATRON wedge path)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.train.run_arm import AdapterModel

    torch.manual_seed(seed)
    backbone = GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=16,
                                          n_layer=2, n_head=2, bos_token_id=0,
                                          eos_token_id=0))
    backbone.config._attn_implementation = "eager"
    mcfg = tiny_mcfg(curriculum=curriculum)
    return AdapterModel(backbone, N_TARGETS, freeze, mcfg, n_value_bins=N_VALUE_BINS)


def scenario_adapter_arms(local: int, out_dir: Path) -> dict:
    """The wedge model under DDP: frozen probe and joint fine-tune (with the curriculum),
    rank 1 never seeing a supervised anchor."""
    from src.train.engine import TrainConfig, train
    from src.train.pretrain import build_optimizer, build_scheduler

    batches = [adapter_batch(600 + i, supervised=i % 2 == 0) for i in range(8)]
    result = {}
    for name, freeze, curriculum in (("frozen", True, "none"),
                                     ("joint", False, "ntp_then_tte")):
        bare = tiny_adapter_model(freeze=freeze, curriculum=curriculum)
        model = _wrap(bare, local)
        tcfg = tiny_tcfg(out_dir / name)
        opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95),
                              trunk_prefixes=("adapter.backbone.",))
        _, manifest = train(model, microbatch_loader(batches, distributed=True), None, opt,
                            build_scheduler(opt, 8, 1),
                            TrainConfig({}, tcfg, {"compile": False}, 8), CPU,
                            vocab_binding=BINDING)
        result[name] = {"updates": manifest.ledger["optimizer_updates"],
                        "state": _state(model)}
    return result


ABLATION_TOKENIZERS = ("fused", "continuous_fused", "textcode")


def scenario_ablation_runner(local: int, out_dir: Path) -> dict:
    """`run_tokenization_ablation.train_arm` for each input representation, with a
    zero-weight head AND a rank that has no supervised anchor."""
    from src.train.pretrain import Loaders
    from src.train.run_tokenization_ablation import ArmRun, TokenizationAblationModel, train_arm

    mcfg = tiny_mcfg(threshold_hazard=0.0)
    batches = [synthetic_batch(300 + i, supervised=i % 2 == 0) for i in range(6)]
    table = torch.Generator().manual_seed(7)
    text_table = torch.randn(VOCAB, 12, generator=table).numpy()
    result = {}
    for tokenizer in ABLATION_TOKENIZERS:
        arm = {"name": tokenizer, "tokenizer": tokenizer, "total_steps": 3, "lr": 1e-2}
        torch.manual_seed(0)
        model = TokenizationAblationModel(
            VOCAB, N_TARGETS, mcfg, arm, n_value_bins=N_VALUE_BINS,
            text_table=text_table if tokenizer == "textcode" else None)
        train_dl = microbatch_loader(batches, distributed=True)
        val_dl = microbatch_loader([synthetic_batch(399)], distributed=False)
        loaders = Loaders(train=train_dl, validation=val_dl, train_dataset=train_dl.dataset,
                          validation_dataset=val_dl.dataset, records=[],
                          data_path=out_dir, value_stats={})
        run = ArmRun(tokenizer, arm, model, loaders, {}, BINDING, N_VALUE_BINS)
        tcfg = tiny_tcfg(out_dir / "unused", val_every=1)
        trained, manifest = train_arm(run, tcfg=tcfg, mcfg=mcfg, device=CPU,
                                      out_dir=out_dir / tokenizer, local=local)
        result[tokenizer] = {"updates": manifest.ledger["optimizer_updates"],
                             "state": _state(trained),
                             "validations": len(manifest.validation)}
    return result


SCENARIOS = {
    "tte_weights_zero": scenario_tte_weights_zero,
    "rank_without_anchor": scenario_rank_without_anchor,
    "accumulation_gradient": scenario_accumulation_gradient,
    "partial_epoch": scenario_partial_epoch,
    "curriculum_warmup": scenario_curriculum_warmup,
    "adapter_arms": scenario_adapter_arms,
    "ablation_runner": scenario_ablation_runner,
}


def _rank_main(scenario: str, out_dir: str) -> None:
    """One rank of a launch (this file run as a script with the torchrun environment)."""
    from src.train.engine import select_device, setup_ddp

    torch.set_num_threads(1)
    local, is_main = setup_ddp(allow_cpu=True)      # the production CPU (gloo) path
    if not dist.is_initialized() or dist.get_backend() != "gloo":
        raise RuntimeError("setup_ddp(allow_cpu=True) did not join a gloo process group")
    if is_main != (local == 0) or select_device(local) != CPU:
        raise RuntimeError("a CPU distributed rehearsal must train on cpu, rank 0 as main")
    try:
        result = SCENARIOS[scenario](local, Path(out_dir))
        torch.save(result, Path(out_dir) / f"rank_{dist.get_rank()}.pt")
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------------ launcher side

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch_two_ranks(scenario: str, out_dir: Path, *, timeout: float = LAUNCH_TIMEOUT_S):
    """Run `scenario` on two gloo ranks. Returns (returncodes, logs, timed_out); a rank
    still alive at the deadline — or left blocked in a collective by a dead peer — is
    killed, so the launch can neither hang nor leak a process."""
    for _ in range(3):
        codes, logs, timed_out = _launch_once(scenario, out_dir, timeout)
        # Another process can take the free port between probing and rank 0 binding it.
        if not any("address already in use" in log.lower() for log in logs):
            break
    return codes, logs, timed_out


def _launch_once(scenario: str, out_dir: Path, timeout: float):
    port = _free_port()
    procs, handles = [], []
    try:
        for rank in range(2):
            env = dict(os.environ)
            env.update({
                "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                "CUDA_VISIBLE_DEVICES": "",   # the CPU path, also on a GPU node
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
                "OMP_NUM_THREADS": "1",
            })
            handle = open(out_dir / f"rank_{rank}.log", "w")
            handles.append(handle)
            procs.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), scenario, str(out_dir)],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + timeout
        peer_grace = None
        while any(p.poll() is None for p in procs) and time.monotonic() < deadline:
            if peer_grace is None and any(p.poll() not in (None, 0) for p in procs):
                # A failed rank leaves its peer waiting in a collective: give the peer a
                # moment to report its own error, then stop waiting.
                peer_grace = time.monotonic() + 10
            if peer_grace is not None and time.monotonic() > peer_grace:
                break
            time.sleep(0.05)
        timed_out = peer_grace is None and any(p.poll() is None for p in procs)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        for handle in handles:
            handle.close()
    logs = [(out_dir / f"rank_{rank}.log").read_text() for rank in range(2)]
    return [p.returncode for p in procs], logs, timed_out


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),
                     "torch.distributed gloo backend is not built on this platform")
class TwoRankTrainingTest(unittest.TestCase):
    def _launch(self, scenario: str, *, keep_all_checkpoints: bool = False) -> list[dict]:
        """Both ranks' results; fails with each rank's log tail when the launch did not
        finish cleanly."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            codes, logs, timed_out = launch_two_ranks(scenario, out_dir)
            if timed_out or any(codes):
                tails = "\n".join(f"--- rank {rank} (exit {code}) ---\n{log[-3000:]}"
                                  for rank, (code, log) in enumerate(zip(codes, logs)))
                self.fail(f"two-rank launch {scenario!r} "
                          f"{'timed out' if timed_out else 'failed'}:\n{tails}")
            results = [torch.load(out_dir / f"rank_{rank}.pt", weights_only=False)
                       for rank in range(2)]
            self.checkpoints = sorted(p.name for p in (out_dir / "ckpt").glob("*.pt"))
            if self.checkpoints:
                self.checkpoint = torch.load(out_dir / "ckpt" / self.checkpoints[-1],
                                             weights_only=False)
            self.checkpoint_by_step = {
                int(p.stem.rsplit("step", 1)[1]): torch.load(p, weights_only=False)
                for p in (out_dir / "ckpt").glob("*.pt")
            } if keep_all_checkpoints else {}
        return results

    def assertSameState(self, a: dict, b: dict, what: str):
        self.assertEqual(a.keys(), b.keys())
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]),
                            f"{what}: {name} differs across ranks "
                            f"(max |d| = {(a[name] - b[name]).abs().max():.3e})")

    def test_launch_is_killed_at_the_deadline_instead_of_hanging(self):
        with tempfile.TemporaryDirectory() as td:
            codes, _, timed_out = launch_two_ranks("partial_epoch", Path(td), timeout=0.05)
        self.assertTrue(timed_out)
        self.assertTrue(all(code not in (None, 0) for code in codes), codes)

    def test_zero_weight_tte_heads_complete_three_updates(self):
        r0, r1 = self._launch("tte_weights_zero")
        self.assertEqual((r0["updates"], r1["updates"]), (3, 3))
        self.assertSameState(r0["state"], r1["state"], "zero-weight TTE heads")
        # Rank 0 alone writes the step-3 checkpoint, and it loads into a bare Model.
        self.assertEqual(self.checkpoints, ["ckpt_ep1_step3.pt"])
        model = build_model(tiny_mcfg(competing_risk=0.0, threshold_hazard=0.0), seed=1)
        model.load_state_dict(self.checkpoint["model"])
        self.assertSameState(dict(model.state_dict()), r0["state"], "checkpoint")
        self.assertEqual(len(self.checkpoint["rng_states"]), 2)

    def test_rank_without_supervised_anchor_completes_updates(self):
        r0, r1 = self._launch("rank_without_anchor")
        self.assertEqual((r0["updates"], r1["updates"]), (3, 3))
        self.assertSameState(r0["state"], r1["state"], "rank without anchors")
        # The heads rank 1 never supervised still trained from rank 0's labels.
        initial = build_model(tiny_mcfg()).state_dict()
        for name in ("cr.fc.weight", "th.mlp.0.weight"):
            self.assertFalse(torch.equal(initial[name], r1["state"][name]), name)

    def test_no_sync_accumulation_matches_single_process_gradient(self):
        r0, r1 = self._launch("accumulation_gradient")
        # Three unsynchronized microsteps, then the synchronized boundary step.
        self.assertEqual(r0["sync"], [False, False, False, True])
        self.assertEqual(r1["sync"], [False, False, False, True])
        self.assertSameState(r0["grad"], r1["grad"], "accumulated gradient")
        reference = accumulated_gradient(
            build_model(tiny_mcfg()),
            microbatch_loader(accumulation_batches(), distributed=False),
            grad_accum=ACCUM_BATCHES)
        self.assertEqual(reference.keys(), r0["grad"].keys())
        for name, expected in reference.items():
            self.assertGreater(float(expected.abs().max()), 0.0, name)
            self.assertTrue(
                torch.allclose(r0["grad"][name], expected, rtol=1e-4, atol=1e-6),
                f"{name}: max |d| = {(r0['grad'][name] - expected).abs().max():.3e}")

    def test_partial_epoch_and_update_limit_keep_ranks_identical(self):
        r0, r1 = self._launch("partial_epoch")
        self.assertEqual((r0["updates"], r1["updates"]), (5, 5))
        # Epoch 1: boundary steps 2 and 4, then the partial 5th microbatch is synchronized;
        # epoch 2 stops at the update limit on a boundary step.
        expected = [False, True, False, True, True, False, True, False, True]
        self.assertEqual(r0["sync"], expected)
        self.assertEqual(r1["sync"], expected)
        self.assertSameState(r0["state"], r1["state"], "partial accumulation")

    def test_curriculum_warm_up_is_ddp_safe_and_ranks_apply_the_same_weights(self):
        from src.train.curriculum import curriculum_weights

        r0, r1 = self._launch("curriculum_warmup", keep_all_checkpoints=True)
        self.assertEqual((r0["updates"], r1["updates"]), (CURRICULUM_STEPS, CURRICULUM_STEPS))
        self.assertSameState(r0["state"], r1["state"], "curriculum")
        # Same weights on the same update on both ranks, one schedule step per optimizer
        # update (two microbatches each), from the engine's update counter.
        expected = [tuple(curriculum_weights(update, CURRICULUM_STEPS)[:4])
                    for update in range(CURRICULUM_STEPS) for _ in range(CURRICULUM_ACCUM)]
        self.assertEqual(r0["weights"], expected)
        self.assertEqual(r1["weights"], expected)
        self.assertEqual(expected[:6], [(1.0, 0.0, 0.0, 0.0)] * 6)      # next-token only
        self.assertEqual(expected[-1], (0.2, 1.0, 1.0, 0.5))
        # Warm-up (updates 0..2) and the first transition update (blend at 0) leave every
        # head exactly as initialised: zero gradient through the touch term, no weight
        # decay. The trunk trains from the first update.
        initial = build_model(tiny_mcfg(curriculum="ntp_then_tte")).state_dict()
        self.assertEqual(sorted(self.checkpoint_by_step), list(range(1, CURRICULUM_STEPS + 1)))
        for step in (1, 2, 3, 4):
            saved = self.checkpoint_by_step[step]["model"]
            for name, value in head_state(saved).items():
                self.assertTrue(torch.equal(value, initial[name]), f"step {step}: {name} moved")
            self.assertFalse(torch.equal(saved["enc.blocks.0.qkv.weight"],
                                         initial["enc.blocks.0.qkv.weight"]))
        # The heads train once their weight is positive (rank 0 supervises them).
        for name in ("cr.fc.weight", "th.mlp.0.weight", "vr.mlp.0.weight"):
            self.assertFalse(torch.equal(r1["state"][name], initial[name]), name)

    def test_adapter_arms_complete_updates_with_skipped_heads(self):
        r0, r1 = self._launch("adapter_arms")
        for name in ("frozen", "joint"):
            with self.subTest(arm=name):
                self.assertEqual((r0[name]["updates"], r1[name]["updates"]), (8, 8))
                self.assertSameState(r0[name]["state"], r1[name]["state"], name)
        frozen = tiny_adapter_model(freeze=True).state_dict()
        for pname, value in r0["frozen"]["state"].items():
            if pname.startswith(("adapter.backbone.", "adapter.next_event.")):
                self.assertTrue(torch.equal(value, frozen[pname]), pname)
        self.assertFalse(torch.equal(r0["frozen"]["state"]["adapter.cr.fc.weight"],
                                     frozen["adapter.cr.fc.weight"]))

    def test_ablation_runner_trains_every_representation(self):
        r0, r1 = self._launch("ablation_runner")
        self.assertEqual(set(r0), set(ABLATION_TOKENIZERS))
        for tokenizer in ABLATION_TOKENIZERS:
            with self.subTest(tokenizer=tokenizer):
                self.assertEqual((r0[tokenizer]["updates"], r1[tokenizer]["updates"]), (3, 3))
                self.assertSameState(r0[tokenizer]["state"], r1[tokenizer]["state"], tokenizer)
                # Epoch-boundary validation runs on rank 0 only; rank 1 waits at the barrier.
                self.assertEqual((r0[tokenizer]["validations"], r1[tokenizer]["validations"]),
                                 (1, 0))


class AdapterObjectiveTest(unittest.TestCase):
    """`CLIFATRONHeads.loss` on the minutes contract (KTD4) with DDP-safe skips (KTD2)."""

    def _prepared(self, batch):
        from src.train.engine import _prepare_batch

        return _prepare_batch(batch, CPU)

    def test_heads_bin_minutes_on_their_own_grids(self):
        from src.model.heads import time_bin

        model = tiny_adapter_model(freeze=True).adapter
        batch = self._prepared(adapter_batch(3))
        # Two anchors: an event at 61 min, and a censoring at 59 min of the 180-min
        # competing-risk bins (48 h / N_TIME_BINS): no full bin observed, so masked out.
        batch["cr_type"] = torch.tensor([0, -1])
        batch["cr_time_min"] = torch.tensor([61, 59])
        batch["cr_mask"] = torch.tensor([True, True])
        out = model.loss(batch)
        H = model.hidden_states(batch["input_ids"], batch["attention_mask"])
        h = H[batch["anchor_batch_idx"], batch["anchor_idx"]]
        expected = model.cr.loss(h[:1], torch.tensor([0]),
                                 time_bin(torch.tensor([61]), N_TIME_BINS, 48))
        self.assertTrue(torch.allclose(out["cr"], expected))
        # An observed-bin count passed as `cr_bin` is no longer read.
        batch["cr_bin"] = torch.tensor([3, 3])
        self.assertTrue(torch.equal(model.loss(batch)["cr"], out["cr"]))

    def test_every_trainable_parameter_gets_a_gradient_whatever_is_skipped(self):
        for freeze in (True, False):
            for supervised in (True, False):
                model = tiny_adapter_model(freeze=freeze)
                batch = self._prepared(adapter_batch(5, supervised=supervised))
                weights = {} if supervised else {"w_cr": 0.0}
                model.adapter.loss(batch, **weights)["total"].backward()
                with self.subTest(freeze=freeze, supervised=supervised):
                    trainable = [(n, p) for n, p in model.named_parameters()
                                 if p.requires_grad]
                    self.assertTrue(trainable)
                    for pname, param in trainable:
                        self.assertIsNotNone(param.grad, pname)
                    # A frozen probe trains only its heads.
                    if freeze:
                        self.assertFalse(any(n.startswith(("adapter.backbone.",
                                                           "adapter.next_event."))
                                             for n, _ in trainable))

    def test_frozen_adapter_refuses_the_curriculum(self):
        with self.assertRaisesRegex(ValueError, "frozen-trunk arm cannot run"):
            tiny_adapter_model(freeze=True, curriculum="ntp_then_tte")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in SCENARIOS:
        _rank_main(sys.argv[1], sys.argv[2])
    else:
        unittest.main()
