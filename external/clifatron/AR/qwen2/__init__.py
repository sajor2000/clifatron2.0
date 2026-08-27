"""
qwen2_sft - Supervised Fine-Tuning with TRL Packing

Efficient SFT training for Qwen2 clinical models using:
- TRL's ConstantLengthDataset for safe packing (prevents attention leakage)
- SFTTrainer for supervised fine-tuning
- Optuna for hyperparameter optimization
- Primary/secondary mode for multi-site vocab locking

Features:
- Global packing: ~8x less padding waste
- Automatic attention masking (no cross-patient leakage)
- HP search: auto-tune learning rate + batch config
- DeepSpeed ZeRO-2/3 support
- W&B logging with site tracking
"""

__version__ = "1.0.0"
