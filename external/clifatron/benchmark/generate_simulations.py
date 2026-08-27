#!/usr/bin/env python3
"""
generate_simulations.py - Generate Clinical Trajectory Simulations

Generate simulated clinical trajectories using trained language models.
For each hospitalization in test data, generates multiple simulated sequences
by continuing from the input sequence.

Usage (Single GPU):
    uv run benchmark/generate_simulations.py \
        --checkpoint models/gpt2_hf/checkpoints/clif-gpt2_hf-tiny/final_model \
        --model-type gpt2_hf \
        --input-dir benchmark/data \
        --n-simulations 10 \
        --max-tokens 5000 \
        --output simulations_gpt2_hf.parquet

Usage (Multi-GPU with torchrun):
    uv run torchrun --nproc_per_node=2 benchmark/generate_simulations.py \
        --checkpoint models/gpt2_hf/checkpoints/clif-gpt2_hf-tiny/final_model \
        --model-type gpt2_hf \
        --input-dir benchmark/data \
        --n-simulations 10 \
        --max-tokens 5000 \
        --output simulations_gpt2_hf.parquet

Features:
    - Supports both gpt2_hf and qwen2 models
    - Filters sequences to only include tokens in model vocabulary
    - Uses KV cache for efficient generation
    - Stops at EOS/SEP tokens or max token limit
    - Generates multiple simulations per hospitalization
    - Privacy-preserving: excludes hospitalization IDs from output
    - Multi-GPU support with torchrun for parallel generation
"""

import os
import sys
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm

# Add AR directories to path for tokenizer import
sys.path.insert(0, str(Path(__file__).parent.parent / 'AR'))


def setup_distributed() -> Tuple[int, int, bool]:
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, is_distributed)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize distributed backend
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)

        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed(is_distributed: bool):
    """Cleanup distributed environment."""
    if is_distributed:
        dist.destroy_process_group()


def load_model_and_tokenizer(checkpoint_path: str, model_type: str, device: str, max_tokens: int = 5000):
    """
    Load trained model and tokenizer based on model type with torch.compile() optimization.

    Args:
        checkpoint_path: Path to checkpoint directory
        model_type: 'gpt2_hf' or 'qwen2'
        device: Device to load model on
        max_tokens: Maximum tokens for static KV cache

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")
    if device == 'cuda' and torch.cuda.is_bf16_supported():
        print("  Using bfloat16 precision for faster GPU inference")

    if model_type == 'gpt2_hf':
        from transformers import GPT2LMHeadModel
        from gpt2_hf.tokenizer.clinical_tokenizer import ClinicalTokenizer

        tokenizer = ClinicalTokenizer.from_pretrained(checkpoint_path)
        model = GPT2LMHeadModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
        )

    elif model_type == 'qwen2':
        from transformers import Qwen2ForCausalLM
        from qwen2.tokenizer.clinical_tokenizer import ClinicalTokenizer

        tokenizer = ClinicalTokenizer.from_pretrained(checkpoint_path)
        model = Qwen2ForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    model.eval()

    # Apply torch.compile() optimization for faster generation
    if device == 'cuda':
        print("  Applying torch.compile() optimization...")
        try:
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
            print("  ✓ torch.compile() applied")
        except Exception as e:
            print(f"  ⚠ torch.compile() failed (continuing without it): {e}")

    print(f"  ✓ Loaded tokenizer (vocab size: {len(tokenizer)})")
    print(f"  ✓ Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print()

    return model, tokenizer


def filter_sequence_by_vocab(sequence: str, vocab: Dict[str, int]) -> str:
    """
    Filter a sequence to only include tokens in the vocabulary.

    Args:
        sequence: Space-separated token sequence
        vocab: Vocabulary dictionary mapping tokens to IDs

    Returns:
        Filtered space-separated token sequence
    """
    tokens = sequence.split()
    filtered_tokens = [token for token in tokens if token in vocab]
    return ' '.join(filtered_tokens)


def load_test_data(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test datasets from parquet files.

    Args:
        input_dir: Directory containing test parquet files

    Returns:
        Tuple of (disposition_df, respiratory_df)
    """
    disposition_path = input_dir / 'task1_task2_disposition_test.parquet'
    respiratory_path = input_dir / 'task3_task4_respiratory_test.parquet'

    disposition_df = None
    respiratory_df = None

    if disposition_path.exists():
        disposition_df = pd.read_parquet(disposition_path)
        print(f"  ✓ Loaded disposition test data: {len(disposition_df)} hospitalizations")

    if respiratory_path.exists():
        respiratory_df = pd.read_parquet(respiratory_path)
        print(f"  ✓ Loaded respiratory test data: {len(respiratory_df)} hospitalizations")

    return disposition_df, respiratory_df


def generate_simulations(
    model,
    tokenizer,
    input_sequence: str,
    n_simulations: int,
    max_tokens: int,
    device: str,
    batch_size: int = 10,
    max_input_length: int = 4000,
) -> List[str]:
    """
    Generate multiple simulations from an input sequence using smaller batched generation.

    Args:
        model: Trained language model
        tokenizer: Clinical tokenizer
        input_sequence: Space-separated input token sequence
        n_simulations: Number of simulations to generate
        max_tokens: Maximum tokens to generate per simulation
        device: Device to use
        batch_size: Number of simulations to generate in each batch (default: 10)
        max_input_length: Maximum input sequence length; truncates from end if longer (default: 4000)

    Returns:
        List of generated sequences (space-separated tokens)
    """
    # Get stop token IDs
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    bos_token_id = tokenizer.bos_token_id

    # Tokenize input and prepend BOS token
    input_ids = tokenizer.encode(input_sequence, add_special_tokens=False)

    # Truncate extremely long inputs to prevent OOM (keep last N tokens as context)
    if len(input_ids) > max_input_length:
        # Keep the most recent tokens (end of sequence) as they're most relevant
        input_ids = input_ids[-max_input_length:]

    input_ids = [bos_token_id] + input_ids  # Add BOS at the beginning
    input_length = len(input_ids)

    # Generate simulations in smaller batches to avoid OOM
    simulations = []
    for batch_start in range(0, n_simulations, batch_size):
        batch_end = min(batch_start + batch_size, n_simulations)
        current_batch_size = batch_end - batch_start

        # Batch current set of simulations
        input_tensor = torch.tensor([input_ids] * current_batch_size, device=device)

        # Generate this batch
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                do_sample=True,  # Sampling for diverse simulations
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[eos_token_id, sep_token_id, bos_token_id],  # Stop at any of these
                use_cache=True,  # Enable KV cache (cleared via model reload every N hospitalizations)
            )

        # Extract generated tokens for each simulation in this batch
        for batch_idx in range(current_batch_size):
            # Extract only the newly generated tokens (exclude input)
            generated_ids = output_ids[batch_idx][input_length:].tolist()
            generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

            # Join tokens with spaces (already in token form)
            generated_sequence = ' '.join(generated_tokens)
            simulations.append(generated_sequence)

        # Clear memory after each batch
        del input_tensor, output_ids
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    return simulations


def main():
    parser = argparse.ArgumentParser(
        description='Generate clinical trajectory simulations using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint directory'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['gpt2_hf', 'qwen2'],
        help='Model type: gpt2_hf or qwen2'
    )

    # Data arguments
    parser.add_argument(
        '--input-dir',
        type=str,
        default='benchmark/data',
        help='Directory containing test parquet files (default: benchmark/data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['disposition', 'respiratory', 'both'],
        default='both',
        help='Which dataset(s) to process (default: both)'
    )

    # Generation arguments
    parser.add_argument(
        '--n-simulations',
        type=int,
        default=10,
        help='Number of simulations per hospitalization (default: 10)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=5000,
        help='Maximum tokens to generate per simulation (default: 5000)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=50,
        help='Save results every N hospitalizations to avoid OOM (default: 50)'
    )
    parser.add_argument(
        '--reload-every',
        type=int,
        default=10,
        help='Reload model every N hospitalizations to clear GPU memory (default: 10)'
    )
    parser.add_argument(
        '--max-input-length',
        type=int,
        default=4000,
        help='Maximum input sequence length (truncates from end if longer) (default: 4000)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file path'
    )

    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of hospitalizations to process (default: all)'
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, is_distributed = setup_distributed()

    # Update device for distributed
    if is_distributed:
        device = f'cuda:{rank}'
    else:
        device = args.device

    # Only rank 0 prints header
    if rank == 0:
        print("=" * 80)
        print("CLINICAL TRAJECTORY SIMULATION GENERATION")
        print("=" * 80)
        print(f"Model: {args.model_type}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Input dir: {args.input_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Simulations per hospitalization: {args.n_simulations}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Save every: {args.save_every} hospitalizations")
        print(f"Reload model every: {args.reload_every} hospitalizations")
        print(f"Output: {args.output}")
        print(f"Device: {device}")
        if is_distributed:
            print(f"Distributed: Yes (rank {rank}/{world_size})")
        print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=device,
        max_tokens=args.max_tokens,
    )

    # Get vocabulary for filtering
    # Handle different tokenizer types
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()  # HuggingFace tokenizer (gpt2_hf)
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab  # ClinicalTokenizer (qwen2)
    else:
        raise AttributeError("Tokenizer has no get_vocab() method or vocab attribute")

    # Load test data
    if rank == 0:
        print("Loading test data...")
    input_dir = Path(args.input_dir)
    disposition_df, respiratory_df = load_test_data(input_dir)
    if rank == 0:
        print()

    # Determine which datasets to process
    datasets_to_process = []
    if args.dataset in ['disposition', 'both'] and disposition_df is not None:
        datasets_to_process.append(('disposition', disposition_df))
    if args.dataset in ['respiratory', 'both'] and respiratory_df is not None:
        datasets_to_process.append(('respiratory', respiratory_df))

    if not datasets_to_process:
        if rank == 0:
            print("ERROR: No datasets to process!")
        cleanup_distributed(is_distributed)
        return

    # Generate simulations for each dataset
    for dataset_name, df in datasets_to_process:
        if rank == 0:
            print(f"\nProcessing {dataset_name} dataset...")
            print(f"  Total hospitalizations: {len(df)}")

        # Map dataset name to task folder
        task_folder = 'task3-task4' if dataset_name == 'respiratory' else 'task1-task2'

        # Limit samples if specified
        if args.max_samples:
            df = df.head(args.max_samples)
            if rank == 0:
                print(f"  Processing first {len(df)} hospitalizations")

        # Shard data across ranks for distributed processing
        if is_distributed:
            # Each rank gets a different slice of the data
            df_shards = [df.iloc[i::world_size] for i in range(world_size)]
            df = df_shards[rank].reset_index(drop=True)
            if rank == 0:
                print(f"  Distributed: Each rank processes ~{len(df)} hospitalizations")

        # Filter sequences by vocabulary
        if rank == 0:
            print("  Filtering sequences by vocabulary...")
        df['filtered_clif_text'] = df['clif_text'].apply(
            lambda x: filter_sequence_by_vocab(x, vocab)
        )

        # Generate simulations
        if rank == 0:
            print(f"  Generating {args.n_simulations} simulations per hospitalization...")
            print(f"  Saving results every {args.save_every} hospitalizations to avoid OOM")

        # Construct output path
        output_path = Path(args.output)
        base_dir = output_path.parent / 'simulations' / task_folder / args.model_type
        base_dir.mkdir(parents=True, exist_ok=True)

        if is_distributed:
            output_stem = output_path.stem
            output_suffix = output_path.suffix
            final_output_path = base_dir / f"{output_stem}_rank{rank}{output_suffix}"
        else:
            final_output_path = base_dir / output_path.name

        # Track state for incremental saving
        dataset_results = []
        total_saved = 0
        simulation_id_offset = rank * 1000000  # Large offset per rank

        # Only show progress bar on rank 0
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"  Rank {rank}", disable=(rank != 0))
        for batch_idx, (idx, row) in enumerate(iterator):
            input_sequence = row['filtered_clif_text']

            if not input_sequence.strip():
                print(f"    WARNING: Empty sequence after filtering for index {idx}, skipping")
                continue

            # Generate simulations
            simulations = generate_simulations(
                model=model,
                tokenizer=tokenizer,
                input_sequence=input_sequence,
                n_simulations=args.n_simulations,
                max_tokens=args.max_tokens,
                device=device,
                max_input_length=args.max_input_length,
            )

            # Create result records (one per simulation)
            for sim_num, generated_seq in enumerate(simulations, start=1):
                result = {
                    'hospitalization_id': row['hospitalization_id'],
                    'simulation_number': sim_num,
                    'generated_sequence': generated_seq,
                    'dataset': dataset_name,
                }

                # Add task-specific labels
                if dataset_name == 'disposition':
                    result['label_home'] = row['label_home']
                    result['label_ltach'] = row['label_ltach']
                    result['disposition'] = row['disposition']
                elif dataset_name == 'respiratory':
                    result['task3_label'] = row['task3_label']
                    result['task4_proportion'] = row['task4_proportion']

                dataset_results.append(result)

            # Clear memory after each hospitalization to prevent OOM
            del simulations
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()

            # Clear model's internal KV cache after each hospitalization
            # This prevents cache accumulation across hospitalizations while keeping use_cache=True for speed
            if hasattr(model, 'past_key_values') and model.past_key_values is not None:
                model.past_key_values = None
            if hasattr(model.config, 'use_cache'):
                # Briefly toggle to flush any cached state
                original_use_cache = model.config.use_cache
                model.config.use_cache = False
                model.config.use_cache = original_use_cache
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Reload model every N hospitalizations to fully clear GPU memory
            if (batch_idx + 1) % args.reload_every == 0:
                if rank == 0:
                    print(f"\n  🔄 Reloading model at {batch_idx + 1} hospitalizations to clear GPU memory...")

                # Delete model and clear all GPU memory
                del model
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()

                # Reload model fresh
                model, tokenizer = load_model_and_tokenizer(
                    checkpoint_path=args.checkpoint,
                    model_type=args.model_type,
                    device=device,
                    max_tokens=args.max_tokens,
                )

                if rank == 0:
                    print(f"  ✓ Model reloaded successfully")

            # Save incrementally every N hospitalizations
            if (batch_idx + 1) % args.save_every == 0 and len(dataset_results) > 0:
                # Convert batch to DataFrame
                batch_df = pd.DataFrame(dataset_results)

                # Add simulation IDs
                start_id = simulation_id_offset + total_saved
                batch_df.insert(0, 'simulation_id', range(start_id, start_id + len(batch_df)))

                # Append to file (create if first time)
                if total_saved == 0:
                    batch_df.to_parquet(final_output_path, index=False)
                else:
                    # Read existing, append, and write back
                    existing_df = pd.read_parquet(final_output_path)
                    combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    combined_df.to_parquet(final_output_path, index=False)

                total_saved += len(batch_df)
                if rank == 0:
                    print(f"\n  Rank {rank}: Saved batch at {batch_idx + 1} hospitalizations ({total_saved} total simulations)")

                # Clear results and memory to free memory
                dataset_results = []
                del batch_df
                if 'existing_df' in locals():
                    del existing_df
                if 'combined_df' in locals():
                    del combined_df
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                gc.collect()

        # Save any remaining results
        if len(dataset_results) > 0:
            batch_df = pd.DataFrame(dataset_results)
            start_id = simulation_id_offset + total_saved
            batch_df.insert(0, 'simulation_id', range(start_id, start_id + len(batch_df)))

            if total_saved == 0:
                batch_df.to_parquet(final_output_path, index=False)
            else:
                existing_df = pd.read_parquet(final_output_path)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_parquet(final_output_path, index=False)

            total_saved += len(batch_df)

        # Read final file to get total count
        results_df = pd.read_parquet(final_output_path)

        if rank == 0:
            print(f"\n  ✓ Final results saved to: {final_output_path}")
            print(f"  Total simulations generated (this rank): {len(results_df)}")
            print(f"  Columns: {list(results_df.columns)}")

    # Cleanup distributed
    cleanup_distributed(is_distributed)

    if rank == 0:
        print()
        print("=" * 80)
        print("GENERATION COMPLETE!")
        print("=" * 80)
        if is_distributed:
            print(f"\nNote: Results saved in {world_size} separate files.")
            print("To combine: Use pandas to concat all _rank*.parquet files")


if __name__ == '__main__':
    main()
