import json
import os
import shutil
import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, step=0, rng_states=None, manifest=None):
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
