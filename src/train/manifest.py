import datetime
import os
import uuid

import torch


class Manifest:
    def __init__(self, model_name, config, seed, ckpt_dir):
        self.run_id = uuid.uuid4().hex[:12]
        self.model_name = model_name
        self.config = config
        self.seed = seed
        self.ckpt_dir = ckpt_dir
        self.lineage_parent = None
        self.env = {}
        self.validation = []
        self.ledger = {}

    def record_env(self):
        self.env = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
        }

    def record_validation(self, epoch, val_loss):
        self.validation.append({"epoch": epoch, "val_loss": val_loss})

    def record_ledger(self, samples, tokens, ntp_tokens, optimizer_step):
        self.ledger = {
            "samples_seen": samples,
            "tokens_seen": tokens,
            "ntp_eligible_tokens": ntp_tokens,
            "optimizer_updates": optimizer_step,
        }

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "config": self.config,
            "seed": self.seed,
            "ckpt_dir": self.ckpt_dir,
            "lineage_parent": self.lineage_parent,
            "env": self.env,
            "validation": self.validation,
            "ledger": self.ledger,
        }