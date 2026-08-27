#!/usr/bin/env python3
"""
Build Benchmark Dataset - Multi-Task Data Preparation

This script prepares benchmark datasets for 3 tasks:
- Task 1 & 2: Shared disposition file (disposition_home and disposition_ltach)
- Task 3: IMV status file (only patients with IMV in 0-24hr ICU)

All sequences are truncated at 24-hour ICU completion.

Output files:
  - task1_task2_disposition_{train_val,test}.parquet
  - task3_imv_status_{train_val,test}.parquet

Usage:
    uv run build_benchmark.py
"""

import argparse
import logging
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add benchmark directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import prepare_benchmark_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to build benchmark dataset."""
    parser = argparse.ArgumentParser(
        description="Build multi-task benchmark datasets (disposition & IMV status)"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="OutputTokens/narratives",
        help="Directory containing narrative sequences",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/data",
        help="Directory to save processed benchmark data (generates 2 files per split)",
    )

    parser.add_argument(
        "--cohort-file",
        type=str,
        default="OutputTokens/tokentables/cohort.parquet",
        help="Path to cohort.parquet file (for 24hr ICU truncation)",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sequence length",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cohort_path = Path(args.cohort_file)

    logger.info("=" * 80)
    logger.info("Building Multi-Task Benchmark Datasets")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cohort file: {cohort_path}")
    logger.info(f"Min sequence length: {args.min_length}")
    logger.info(f"Max sequence length: {args.max_length}")
    logger.info("")
    logger.info("Output Files:")
    logger.info("  - task1_task2_disposition_{train_val,test}.parquet")
    logger.info("  - task3_imv_status_{train_val,test}.parquet")

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Validate cohort file
    if not cohort_path.exists():
        logger.error(f"Cohort file does not exist: {cohort_path}")
        sys.exit(1)

    # Disposition tokens to extract
    disposition_tokens = [
        "disposition_home",
        "disposition_expired",
        "disposition_rehab",
        "disposition_snf",
        "disposition_hospice",
        "disposition_ltach",
        "disposition_other",
    ]

    # Prepare benchmark data with task-specific files
    logger.info("Preparing multi-task benchmark data with 24hr ICU truncation...")
    logger.info("This will generate 2 files:")
    logger.info("  1. Shared disposition file (Tasks 1 & 2)")
    logger.info("  2. Respiratory status file (Tasks 3 & 4 - IMV patients only)")
    try:
        prep_config = {
            "positive_class": "disposition_home",  # Default for stats only
            "disposition_tokens": disposition_tokens,
            "min_sequence_length": args.min_length,
            "max_sequence_length": args.max_length,
            "truncate_at_24hr_icu": True,
        }

        stats = prepare_benchmark_data(
            input_dir=input_dir,
            output_dir=output_dir,
            vocab=None,  # Vocabulary not needed for benchmark generation
            cohort_path=cohort_path,
            config=prep_config,
        )

        logger.info("Data preparation completed successfully!")

        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info("-" * 80)

        if "train_val" in stats:
            logger.info("\nTrain/Val Split:")

            if "task1_home" in stats["train_val"]:
                logger.info("\n  Task 1 - Discharged Home:")
                task1 = stats["train_val"]["task1_home"]
                logger.info(f"    Total examples:     {task1['total_examples']}")
                logger.info(
                    f"    Positive examples:  {task1['positive_examples']} "
                    f"({task1['positive_ratio']:.2%})"
                )
                logger.info(f"    Negative examples:  {task1['negative_examples']}")

            if "task2_ltach" in stats["train_val"]:
                logger.info("\n  Task 2 - LTACH:")
                task2 = stats["train_val"]["task2_ltach"]
                logger.info(f"    Total examples:     {task2['total_examples']}")
                logger.info(
                    f"    Positive examples:  {task2['positive_examples']} "
                    f"({task2['positive_ratio']:.2%})"
                )
                logger.info(f"    Negative examples:  {task2['negative_examples']}")

            if "task3_imv" in stats["train_val"]:
                logger.info("\n  Task 3 - IMV Status (IMV patients only):")
                task3 = stats["train_val"]["task3_imv"]
                logger.info(f"    Total examples:     {task3['total_examples']}")
                if "label_counts" in task3:
                    logger.info(f"    Label distribution: {task3['label_counts']}")

            if "task4_hypoxic" in stats["train_val"]:
                logger.info("\n  Task 4 - Hypoxic Failure Proportion (IMV patients only):")
                task4 = stats["train_val"]["task4_hypoxic"]
                logger.info(f"    Total examples:     {task4['total_examples']}")
                logger.info(f"    Mean proportion:    {task4['mean_proportion']:.3f}")
                logger.info(f"    Min proportion:     {task4['min_proportion']:.3f}")
                logger.info(f"    Max proportion:     {task4['max_proportion']:.3f}")

        if "test" in stats:
            logger.info("\nTest Split:")

            if "task1_home" in stats["test"]:
                logger.info("\n  Task 1 - Discharged Home:")
                task1 = stats["test"]["task1_home"]
                logger.info(f"    Total examples:     {task1['total_examples']}")
                logger.info(
                    f"    Positive examples:  {task1['positive_examples']} "
                    f"({task1['positive_ratio']:.2%})"
                )
                logger.info(f"    Negative examples:  {task1['negative_examples']}")

            if "task2_ltach" in stats["test"]:
                logger.info("\n  Task 2 - LTACH:")
                task2 = stats["test"]["task2_ltach"]
                logger.info(f"    Total examples:     {task2['total_examples']}")
                logger.info(
                    f"    Positive examples:  {task2['positive_examples']} "
                    f"({task2['positive_ratio']:.2%})"
                )
                logger.info(f"    Negative examples:  {task2['negative_examples']}")

            if "task3_imv" in stats["test"]:
                logger.info("\n  Task 3 - IMV Status (IMV patients only):")
                task3 = stats["test"]["task3_imv"]
                logger.info(f"    Total examples:     {task3['total_examples']}")
                if "label_counts" in task3:
                    logger.info(f"    Label distribution: {task3['label_counts']}")

            if "task4_hypoxic" in stats["test"]:
                logger.info("\n  Task 4 - Hypoxic Failure Proportion (IMV patients only):")
                task4 = stats["test"]["task4_hypoxic"]
                logger.info(f"    Total examples:     {task4['total_examples']}")
                logger.info(f"    Mean proportion:    {task4['mean_proportion']:.3f}")
                logger.info(f"    Min proportion:     {task4['min_proportion']:.3f}")
                logger.info(f"    Max proportion:     {task4['max_proportion']:.3f}")

        # Save statistics
        stats_file = output_dir / "dataset_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(
                {
                    "statistics": stats,
                    "configuration": prep_config,
                    "disposition_tokens": disposition_tokens,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        logger.info(f"\nStatistics saved to {stats_file}")

        logger.info("\n" + "=" * 80)
        logger.info("Benchmark datasets built successfully!")
        logger.info(f"Processed data saved to: {output_dir}")
        logger.info("")
        logger.info("Generated files:")
        logger.info("  - task1_task2_disposition_{train_val,test}.parquet (for Tasks 1 & 2)")
        logger.info("  - task3_task4_respiratory_{train_val,test}.parquet (for Tasks 3 & 4)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to prepare benchmark data: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
