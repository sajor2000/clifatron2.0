import os
import shutil
import tempfile
from pathlib import Path

import torch

from src.data.segments import check_binding


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, step=0, rng_states=None,
                    manifest=None, vocab_binding=None):
    """Atomically write a training checkpoint.

    `vocab_binding` (`segments.artifact_binding(vocab.json)`) records the tokenizer
    version, vocabulary and segments the model was trained on (KTD7); generation, the
    viewer and training resume refuse a checkpoint bound to anything else."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="ckpt_", suffix=".pt", delete=False) as tf:
        tmp = tf.name
    try:
        ckpt = {
            "schema_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "rng_states": rng_states or {},
            "manifest": manifest.to_dict() if manifest else {},
            "vocab_binding": dict(vocab_binding) if vocab_binding else None,
        }
        torch.save(ckpt, tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        shutil.move(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("schema_version", 0) < 1:
        raise ValueError("Incompatible checkpoint schema version")
    return ckpt


def verify_checkpoint_binding(ckpt, vocab_blob) -> None:
    """Refuse a checkpoint whose recorded vocabulary/segments binding is not
    `vocab_blob`'s, or that records none (trained before tokenizer v2)."""
    check_binding(ckpt.get("vocab_binding"), vocab_blob, what="checkpoint")
