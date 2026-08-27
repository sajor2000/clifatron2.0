#!/bin/bash

# CLIF GPT2 Training Pipeline
# This script runs all steps of the pipeline sequentially

set -e  # Exit on error

echo "======================================"
echo "CLIF GPT2 Training Pipeline"
echo "======================================"
echo ""

# Check if input file is provided
INPUT_FILE=${1:-"clif_sentences.parquet"}
MODEL_SIZE=${2:-"small"}

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    echo "Usage: ./run_pipeline.sh [path/to/clif_sentences.parquet] [model_size]"
    echo "  model_size: small, medium, large, xl (default: small)"
    exit 1
fi

# Validate model size
if [[ ! "$MODEL_SIZE" =~ ^(small|medium|large|xl)$ ]]; then
    echo "Error: Invalid model size '$MODEL_SIZE'"
    echo "Model size must be one of: small, medium, large, xl"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "Model size: $MODEL_SIZE"
echo ""

# Step 1: Prepare data
echo "Step 1/4: Preparing data..."
uv run 01_prepare_data.py --input "$INPUT_FILE"
echo "✓ Data preparation complete"
echo ""

# Step 2: Build vocabulary
echo "Step 2/4: Building vocabulary..."
uv run 02_build_vocab.py
echo "✓ Vocabulary building complete"
echo ""

# Step 3: Create splits
echo "Step 3/4: Creating train/val/test splits..."
uv run 03_create_splits.py
echo "✓ Data splitting complete"
echo ""

# Step 4: Train model
echo "Step 4/4: Training GPT2 model ($MODEL_SIZE)..."
uv run 04_train_gpt2.py --model-size "$MODEL_SIZE" --epochs 1
echo "✓ Model training complete"
echo ""

echo "======================================"
echo "Pipeline completed successfully!"
echo "======================================"
echo ""
echo "Model saved to: ./gpt2_output/models/clif-gpt2-$MODEL_SIZE/final_model/"
echo ""
echo "To use the model:"
echo "  - Model config: ./gpt2_output/models/clif-gpt2-$MODEL_SIZE/final_model/config.json"
echo "  - Model weights: ./gpt2_output/models/clif-gpt2-$MODEL_SIZE/final_model/pytorch_model.bin"
echo "  - Vocabulary: ./gpt2_output/models/clif-gpt2-$MODEL_SIZE/final_model/vocab.gzip"
