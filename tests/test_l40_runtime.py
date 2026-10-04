"""L40 GPU / runtime fixes (2x L40, torchrun, bf16) that run without a GPU.

- H1: every encoder arm keeps an fp32 residual stream under bf16 autocast (the TextCode
  arm's input is a Linear, which autocast runs in bf16; the fused arms start from an
  nn.Embedding, which it leaves in fp32);
- L1: RoPE angles stay fp32 but cos/sin reach q/k in q's dtype;
- M7: NCCL init binds the device first and passes device_id and a configurable timeout;
- M1: the validation interval scales with the run;
- M2: a repeated stop signal right after the first is ignored, SIGHUP is handled, and two
  SIGINTs to a single-process run still produce one checkpoint;
- M3: host syncs per microbatch;
- M4: threshold_eval defaults to CUDA when present;
- M5/M6: pre-flight NCCL and driver checks;
- L3: TextCode's frozen table is not broadcast by DDP every forward.

All data is synthetic.
"""

import datetime as dt
import os
import signal
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

from src.train import engine

try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
    from test_ddp_multihead import (BINDING, N_TARGETS, N_VALUE_BINS, VOCAB, build_model,
                                    synthetic_batch, tiny_mcfg, tiny_tcfg)
except ImportError:  # pragma: no cover - run from the repo root as a package
    from tests.test_ddp_multihead import (BINDING, N_TARGETS, N_VALUE_BINS, VOCAB,
                                          build_model, synthetic_batch, tiny_mcfg, tiny_tcfg)

CPU = torch.device("cpu")


def _encoders():
    """One instance of every input representation's encoder (all share CLIFEncoder's
    trunk)."""
    from src.model.encoder import CLIFEncoder
    from src.model.encoder_continuous import ContinuousFusedEncoder
    from src.model.encoder_textcode import TextCodeEncoder
    from src.train.pretrain import OrderOnlyEncoder

    mcfg = tiny_mcfg()
    table = np.random.default_rng(0).standard_normal((VOCAB, 12)).astype(np.float32)
    torch.manual_seed(0)
    return {
        "fused": CLIFEncoder(VOCAB, mcfg),
        "order_only": OrderOnlyEncoder(VOCAB, mcfg),
        "continuous_fused": ContinuousFusedEncoder(VOCAB, mcfg),
        "textcode": TextCodeEncoder(VOCAB, mcfg, table),
    }


class ResidualStreamDtypeTest(unittest.TestCase):
    """H1: before the fix the TextCode arm's residual stream was bf16 under autocast."""

    def test_every_encoder_arm_keeps_an_fp32_residual_stream_under_bf16_autocast(self):
        batch = synthetic_batch(1)
        for name, enc in _encoders().items():
            with self.subTest(arm=name):
                seen = []
                hooks = [blk.register_forward_pre_hook(
                    lambda _m, args: seen.append(args[0].dtype)) for blk in enc.blocks]
                hooks.append(enc.ln_f.register_forward_pre_hook(
                    lambda _m, args: seen.append(args[0].dtype)))
                kwargs = {}
                if name == "continuous_fused":
                    kwargs = {"continuous_value": batch["input_value"],
                              "continuous_value_mask": batch["input_value_mask"]}
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    enc(batch["input_ids"], batch["pos_min"], **kwargs)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(len(seen), len(enc.blocks) + 1)
                self.assertTrue(all(d == torch.float32 for d in seen), (name, seen))

    def test_soft_token_textcode_input_is_fp32_too(self):
        enc = _encoders()["textcode"]
        token = torch.randint(1, VOCAB, (2, 5, 2))
        weight = torch.full((2, 5, 2), 0.5)
        seen = []
        enc.blocks[0].register_forward_pre_hook(lambda _m, args: seen.append(args[0].dtype))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            enc(token, torch.arange(5).repeat(2, 1), weight)
        self.assertEqual(seen, [torch.float32])


class RopeDtypeTest(unittest.TestCase):
    """L1: angles in fp32 (test_rope_base), cos/sin applied in q's dtype."""

    def test_rope_cache_is_fp32_and_rotation_keeps_the_input_dtype(self):
        from src.model.encoder import apply_rope, build_rope_cache

        pos = torch.tensor([[0, 10_000, 500_000]])
        cos, sin = build_rope_cache(pos, 8, 10000.0)
        self.assertEqual((cos.dtype, sin.dtype), (torch.float32, torch.float32))
        q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
        out = apply_rope(q, cos, sin)
        self.assertEqual(out.dtype, torch.bfloat16)
        reference = apply_rope(q.float(), cos, sin)
        torch.testing.assert_close(out.float(), reference, atol=0.05, rtol=0.02)


class NcclInitOrderTest(unittest.TestCase):
    """M7: `torch.cuda.set_device(local)` before `init_process_group("nccl", device_id=...,
    timeout=...)`; the timeout is configurable (default 2 h, not NCCL's 10 min)."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1",
                                               "WORLD_SIZE": "2"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(engine.DDP_TIMEOUT_ENV, None)

    def _setup(self):
        calls = []
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.set_device",
                           side_effect=lambda d: calls.append(("set_device", d))), \
                mock.patch.object(engine.dist, "init_process_group",
                                  side_effect=lambda *a, **k: calls.append(("init", a, k))), \
                mock.patch.object(engine.dist, "get_rank", return_value=1):
            self.assertEqual(engine.setup_ddp(), (1, False))
        return calls

    def test_device_is_bound_before_the_nccl_group_with_device_id_and_timeout(self):
        calls = self._setup()
        self.assertEqual([c[0] for c in calls], ["set_device", "init"])
        self.assertEqual(calls[0][1], 1)
        _, args, kwargs = calls[1]
        self.assertEqual(args, ("nccl",))
        self.assertEqual(kwargs["device_id"], torch.device("cuda", 1))
        self.assertEqual(kwargs["timeout"], dt.timedelta(hours=2))

    def test_timeout_is_configurable_from_the_environment(self):
        os.environ[engine.DDP_TIMEOUT_ENV] = "30"
        _, _, kwargs = self._setup()[1]
        self.assertEqual(kwargs["timeout"], dt.timedelta(minutes=30))

    def test_bad_timeout_is_refused(self):
        os.environ[engine.DDP_TIMEOUT_ENV] = "0"
        with self.assertRaisesRegex(ValueError, engine.DDP_TIMEOUT_ENV):
            engine.ddp_timeout()


class ValidationScheduleTest(unittest.TestCase):
    """M1: a fixed val_every of 2000 never validated a 9-300-update run."""

    def _tcfg(self, **eval_schedule):
        tcfg = {"schedule": {}, "runtime": {}}
        if eval_schedule:
            tcfg["eval_schedule"] = eval_schedule
        return tcfg

    def test_interval_scales_with_the_run(self):
        for total in (9, 15, 300, 60_000):
            with self.subTest(total=total):
                s = engine.resolve_schedule(self._tcfg(), total)
                self.assertEqual(s.val_every,
                                 max(1, total // engine.DEFAULT_VALIDATIONS_PER_RUN))
                self.assertLessEqual(s.val_every, total)
                self.assertEqual(s.val_source, "validations_per_run")

    def test_validations_per_run_and_absolute_interval(self):
        s = engine.resolve_schedule(self._tcfg(validations_per_run=4), 100)
        self.assertEqual((s.val_every, s.val_source), (25, "validations_per_run"))
        s = engine.resolve_schedule(self._tcfg(val_every=7), 100)
        self.assertEqual((s.val_every, s.val_source), (7, "val_every"))
        with self.assertRaises(engine.ScheduleError):
            engine.resolve_schedule(self._tcfg(validations_per_run=0), 100)

    def test_train_config_without_eval_schedule_validates_within_a_short_run(self):
        tcfg = tiny_tcfg("unused")
        tcfg.pop("eval_schedule")
        cfg = engine.TrainConfig({}, tcfg, tiny_mcfg(), total_steps=12)
        self.assertLessEqual(cfg.val_every, 12)

    def test_shipped_train_config_validates_screening_runs(self):
        import yaml

        tcfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                               / "configs/train.yaml").read_text())
        for total in (9, 15, 300):
            self.assertLessEqual(engine.resolve_schedule(tcfg, total).val_every, total)


class StopSignalTest(unittest.TestCase):
    """M2: torchrun re-sends the SIGINT of a Ctrl-C; the repeat must not abort."""

    def test_sighup_is_handled(self):
        self.assertIn(signal.SIGHUP, engine.StopRequest.SIGNALS)

    def test_a_repeat_within_the_grace_interval_is_ignored(self):
        stop = engine.StopRequest()
        stop._handle(signal.SIGINT, None)
        stop._handle(signal.SIGINT, None)      # torchrun's forwarded copy
        self.assertTrue(stop.requested)

    def test_a_later_repeat_still_aborts(self):
        stop = engine.StopRequest()
        stop._handle(signal.SIGINT, None)
        stop._first_at -= engine.StopRequest.REPEAT_GRACE_S + 1
        with self.assertRaises(KeyboardInterrupt):
            stop._handle(signal.SIGINT, None)

    def test_two_sigints_to_a_single_process_run_write_one_checkpoint(self):
        class TwoSigints(torch.utils.data.Dataset):
            def __init__(self):
                self.batches = [synthetic_batch(40 + i) for i in range(10)]
                self.sent = False

            def __len__(self):
                return len(self.batches)

            def __getitem__(self, index):
                if index == 3 and not self.sent:
                    self.sent = True
                    os.kill(os.getpid(), signal.SIGINT)
                    os.kill(os.getpid(), signal.SIGINT)
                return self.batches[index]

        from src.train.pretrain import build_optimizer, build_scheduler

        with tempfile.TemporaryDirectory() as td:
            model = build_model(tiny_mcfg())
            opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
            tcfg = tiny_tcfg(Path(td) / "ckpt", ckpt_every=1000)
            loader = torch.utils.data.DataLoader(TwoSigints(), batch_size=None)
            before = signal.getsignal(signal.SIGINT)
            _, manifest = engine.train(
                model, loader, None, opt, build_scheduler(opt, 100, 1),
                engine.TrainConfig({}, tcfg, tiny_mcfg(), 100), CPU, vocab_binding=BINDING)
            files = sorted(p.name for p in (Path(td) / "ckpt").glob("*.pt"))
            self.assertIs(signal.getsignal(signal.SIGINT), before)
        self.assertEqual(manifest.ledger["optimizer_updates"], 4)
        self.assertEqual(files, ["ckpt_ep0_step4.pt"])


class _SyncCounter:
    """Counts explicit device->host reads (`item`, `tolist`, `bool`, `int`, `float`,
    `nonzero`, `unique`) while active. On CUDA each one stalls the stream."""

    NAMES = ("item", "tolist", "__bool__", "__int__", "__float__", "nonzero", "unique")

    def __init__(self):
        self.count = 0
        self._patches = []

    def __enter__(self):
        for name in self.NAMES:
            original = getattr(torch.Tensor, name)

            def counted(*args, _original=original, **kwargs):
                self.count += 1
                return _original(*args, **kwargs)

            patcher = mock.patch.object(torch.Tensor, name, counted)
            patcher.start()
            self._patches.append(patcher)
        return self

    def __exit__(self, *exc):
        for patcher in self._patches:
            patcher.stop()
        return False


class HostSyncTest(unittest.TestCase):
    """M3: explicit host syncs per training microbatch (forward, finiteness check, token
    counts). Before the fix: one per head mask check, one per loss in the finiteness
    check plus the agreed flag, two token counts per microbatch, a per-row loop over
    packed documents, and the threshold head's range check."""

    def _epoch_syncs(self, batches):
        from src.train.pretrain import build_optimizer

        model = build_model(tiny_mcfg())
        opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        with tempfile.TemporaryDirectory() as td:
            # grad_accum above the batch count and max_updates 0: microbatches only
            # (forward, checks, backward, counts), no optimizer step.
            cfg = engine.TrainConfig({}, tiny_tcfg(td, grad_accum=8), tiny_mcfg(), 1)
            with _SyncCounter() as counter:
                engine._train_one_epoch(model, batches, opt, scheduler, 0, cfg, CPU,
                                        rank=0, max_updates=0)
        return counter.count

    def test_a_microbatch_reads_the_device_at_most_a_few_times(self):
        """Per microbatch on CPU: the two head masks' row indices, the threshold head's
        range check (a device-side assert on CUDA, no read) and the agreed finiteness
        flag. It was 11 before the fix. Token counts are read once per epoch."""
        one = self._epoch_syncs([synthetic_batch(5)])
        three = self._epoch_syncs([synthetic_batch(5), synthetic_batch(6), synthetic_batch(7)])
        per_microbatch = (three - one) / 2
        self.assertLessEqual(per_microbatch, 4, (one, three))
        self.assertLessEqual(one - per_microbatch, 2, (one, three))   # epoch-end counts

    def test_packed_document_check_does_not_loop_over_rows(self):
        batch = synthetic_batch(6, n=8)
        batch["document_ids"] = torch.zeros(8, 6, dtype=torch.long)
        model = build_model(tiny_mcfg())
        prepared = engine._prepare_batch(batch, CPU)
        with _SyncCounter() as counter:
            model(prepared)
        with_docs = counter.count
        prepared.pop("document_ids")
        with _SyncCounter() as counter:
            model(prepared)
        self.assertLessEqual(with_docs - counter.count, 1)


class ThresholdEvalDeviceTest(unittest.TestCase):
    """M4: scoring on the L40 node used the CPU unless --device cuda was passed."""

    def test_default_device_is_cuda_when_available(self):
        from src.eval import threshold_eval

        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(threshold_eval.default_device(), "cuda")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(threshold_eval.default_device(), "cpu")


class PreflightRuntimeChecksTest(unittest.TestCase):
    """M5 / M6: one NCCL library, and a driver new enough for torch's CUDA build."""

    def test_required_driver_per_cuda_major(self):
        from src.train import preflight as pf

        self.assertEqual(pf.required_driver_major("13.0"), 580)
        self.assertEqual(pf.required_driver_major("12.6"), 525)
        self.assertIsNone(pf.required_driver_major(None))

    def test_driver_check_fails_below_the_build_requirement(self):
        from src.train import preflight as pf

        bad = pf.driver_requirement_check("570.124.06", "13.0")
        self.assertEqual(bad.status, pf.FAIL)
        self.assertIn("580", bad.detail)
        self.assertIn("cu126", bad.detail)
        self.assertEqual(pf.driver_requirement_check("580.95.05", "13.0").status, pf.PASS)
        self.assertEqual(pf.driver_requirement_check("570.124.06", "12.6").status, pf.PASS)
        self.assertEqual(pf.driver_requirement_check(None, "13.0").status, pf.FAIL)

    def test_nccl_check_fails_on_two_nccl_wheels(self):
        from src.train import preflight as pf

        with mock.patch.object(pf, "installed_nccl_wheels",
                               return_value=["nvidia-nccl-cu12", "nvidia-nccl-cu13"]):
            check = pf.nccl_check(version=(2, 29, 7))
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("nvidia-nccl-cu12", check.detail)
        with mock.patch.object(pf, "installed_nccl_wheels",
                               return_value=["nvidia-nccl-cu13"]):
            check = pf.nccl_check(version=(2, 29, 7))
        self.assertEqual(check.status, pf.PASS)
        self.assertIn("2.29.7", check.detail)

    def test_gpu_checks_include_the_new_checks(self):
        from src.train import preflight as pf

        names = {c.name for c in pf.gpu_checks(skip_gpu=True)}
        self.assertIn("gpu: driver vs CUDA build", names)
        self.assertIn("gpu: NCCL", names)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "no gloo")
class TextCodeBufferBroadcastTest(unittest.TestCase):
    """L3: the frozen TextCode table is a constant; DDP must not broadcast it before every
    forward (it still syncs at construction, and checkpoints still carry it)."""

    def test_wrap_ddp_turns_off_forward_buffer_sync_for_the_textcode_arm(self):
        from src.train.run_tokenization_ablation import TokenizationAblationModel

        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}",
                                rank=0, world_size=1)
        try:
            table = np.random.default_rng(0).standard_normal((VOCAB, 12)).astype(np.float32)
            mcfg = tiny_mcfg()
            arms = {}
            for tokenizer in ("fused", "textcode"):
                arm = {"name": tokenizer, "tokenizer": tokenizer, "total_steps": 3,
                       "lr": 1e-2}
                model = TokenizationAblationModel(
                    VOCAB, N_TARGETS, mcfg, arm, n_value_bins=N_VALUE_BINS,
                    text_table=table if tokenizer == "textcode" else None)
                arms[tokenizer] = engine.wrap_ddp(model, CPU, 0)
            self.assertFalse(engine.ddp_syncs_buffers(arms["textcode"]))
            self.assertTrue(engine.ddp_syncs_buffers(arms["fused"]))
            self.assertIn("enc.text_table", arms["textcode"].module.state_dict())
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
