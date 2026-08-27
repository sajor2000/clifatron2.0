#!/usr/bin/env python3
"""
Simple script to train XGBoost on pre-computed embeddings for all models.
This bypasses the complex embedding extraction logic.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "benchmark" / "embeddings" / "task1_task2_disposition"
DATA_DIR = PROJECT_ROOT / "benchmark" / "data"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "task1-discharged-home" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models to evaluate
MODELS = ["gpt2", "gpt2_hf", "qwen2", "qwen2optuna"]

def load_embeddings(model_type, split):
    """Load pre-computed embeddings and labels from parquet."""
    # Load embeddings
    embedding_file = EMBEDDINGS_DIR / f"embeddings_{model_type}_mean_{split}.npz"
    print(f"Loading embeddings from {embedding_file}...")
    data = np.load(embedding_file)
    embeddings = data['embeddings']
    example_ids = data['example_ids']

    # Load labels from parquet
    parquet_file = DATA_DIR / f"task1_task2_disposition_{split}.parquet"
    print(f"Loading labels from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # Create label mapping from hospitalization_id (task 1 uses label_home)
    id_to_label = dict(zip(df['hospitalization_id'], df['label_home']))
    labels = np.array([id_to_label[eid] for eid in example_ids])

    return embeddings, labels

def train_and_evaluate_xgboost(model_type):
    """Train XGBoost on embeddings and evaluate."""
    print(f"\n{'='*80}")
    print(f"Processing {model_type}")
    print(f"{'='*80}")

    # Load embeddings
    X_train, y_train = load_embeddings(model_type, "train_val")
    X_test, y_test = load_embeddings(model_type, "test")

    print(f"Train shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test shape: {X_test.shape}, labels: {y_test.shape}")
    print(f"Positive rate (train): {y_train.mean():.3f}")
    print(f"Positive rate (test): {y_test.mean():.3f}")

    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    # Train XGBoost
    print("Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    clf.fit(X_train, y_train)

    # Predict
    print("Making predictions...")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_test, y_pred_proba)),
        "auprc": float(average_precision_score(y_test, y_pred_proba))
    }

    # Print metrics
    print(f"\nResults for {model_type}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save results
    results = {
        "method": "embedding_xgboost",
        "model_type": model_type,
        "layer": "mean",
        "classifier": "xgboost",
        "metrics": metrics,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_train_examples": int(len(y_train)),
            "num_test_examples": int(len(y_test)),
            "embedding_dim": int(X_train.shape[1]),
            "xgboost_params": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "scale_pos_weight": float(scale_pos_weight)
            }
        }
    }

    output_file = RESULTS_DIR / f"task1_{model_type}_method1_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    return results

def main():
    """Train and evaluate all models."""
    all_results = {}

    for model_type in MODELS:
        try:
            results = train_and_evaluate_xgboost(model_type)
            all_results[model_type] = results
        except Exception as e:
            print(f"ERROR processing {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary_file = RESULTS_DIR / "task1_method1_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"All results saved to {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
