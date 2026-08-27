#!/usr/bin/env python3
"""
generate_simulations_cpu_optimized.py - CPU-Optimized Clinical Trajectory Simulations

CPU-optimized version using multiprocessing and multithreading for faster generation.

Usage:
    uv run benchmark/generate_simulations_cpu_optimized.py \
        --checkpoint models/qwen2/model_weights \
        --model-type qwen2 \
        --input-dir benchmark/data \
        --n-simulations 20 \
        --max-tokens 5000 \
        --output simulations_qwen2_task34_full_cpu.parquet \
        --num-workers 4

Features:
    - Multiprocessing: Parallel processing across CPU cores
    - Multithreading: Optimized thread count per worker
    - torch.inference_mode(): Faster CPU inference
    - Shared vocabulary filtering
    - Batched generation per worker
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial

import torch
import pandas as pd
from tqdm import tqdm

# Add AR directories to path for tokenizer import
sys.path.insert(0, str(Path(__file__).parent.parent / 'AR'))


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


def load_model_and_tokenizer(checkpoint_path: str, model_type: str, num_threads: int = 4, quantize: bool = False):
    """
    Load trained model and tokenizer for CPU inference.

    Args:
        checkpoint_path: Path to checkpoint directory
        model_type: 'gpt2_hf' or 'qwen2'
        num_threads: Number of threads for this worker
        quantize: Use INT8 dynamic quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    # Set number of threads for this worker
    torch.set_num_threads(num_threads)

    if model_type == 'gpt2_hf':
        from transformers import GPT2LMHeadModel
        from gpt2_hf.tokenizer.clinical_tokenizer import ClinicalTokenizer

        tokenizer = ClinicalTokenizer.from_pretrained(checkpoint_path)
        model = GPT2LMHeadModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
        )

    elif model_type == 'qwen2':
        from transformers import Qwen2ForCausalLM
        from qwen2.tokenizer.clinical_tokenizer import ClinicalTokenizer

        tokenizer = ClinicalTokenizer.from_pretrained(checkpoint_path)
        model = Qwen2ForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to('cpu')
    model.eval()

    # Apply INT8 dynamic quantization if requested (CPU only)
    if quantize:
        print(f"  Applying INT8 dynamic quantization...")
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        print(f"  ✓ INT8 quantization applied")

    return model, tokenizer


def generate_simulations(
    model,
    tokenizer,
    input_sequence: str,
    n_simulations: int,
    max_tokens: int,
) -> List[str]:
    """
    Generate multiple simulations from an input sequence.

    Args:
        model: Trained language model
        tokenizer: Clinical tokenizer
        input_sequence: Space-separated input token sequence
        n_simulations: Number of simulations to generate
        max_tokens: Maximum tokens to generate per simulation

    Returns:
        List of generated sequences (space-separated tokens)
    """
    # Get stop token IDs
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    bos_token_id = tokenizer.bos_token_id

    # Tokenize input and prepend BOS token
    input_ids = tokenizer.encode(input_sequence, add_special_tokens=False)
    input_ids = [bos_token_id] + input_ids
    input_length = len(input_ids)

    # Batch all simulations
    input_tensor = torch.tensor([input_ids] * n_simulations, device='cpu')

    # Generate all simulations in a single batched call with inference_mode
    with torch.inference_mode():  # CPU optimization
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[eos_token_id, sep_token_id, bos_token_id],
            use_cache=True,
        )

    # Extract generated tokens for each simulation
    simulations = []
    for batch_idx in range(n_simulations):
        generated_ids = output_ids[batch_idx][input_length:].tolist()
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        generated_sequence = ' '.join(generated_tokens)
        simulations.append(generated_sequence)

    return simulations


def worker_process(
    worker_id: int,
    df_chunk: pd.DataFrame,
    checkpoint_path: str,
    model_type: str,
    n_simulations: int,
    max_tokens: int,
    num_threads: int,
    dataset_name: str,
    quantize: bool = False,
) -> List[Dict]:
    """
    Worker process that generates simulations for a chunk of data.

    Args:
        worker_id: Worker process ID
        df_chunk: DataFrame chunk to process
        checkpoint_path: Path to model checkpoint
        model_type: 'gpt2_hf' or 'qwen2'
        n_simulations: Number of simulations per hospitalization
        max_tokens: Maximum tokens to generate
        num_threads: Number of threads for this worker
        dataset_name: 'disposition' or 'respiratory'
        quantize: Use INT8 dynamic quantization

    Returns:
        List of result dictionaries
    """
    # Load model and tokenizer for this worker
    print(f"Worker {worker_id}: Loading model...")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, model_type, num_threads, quantize)

    # Get vocabulary
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    else:
        raise AttributeError("Tokenizer has no get_vocab() method or vocab attribute")

    # Filter sequences by vocabulary
    df_chunk['filtered_clif_text'] = df_chunk['clif_text'].apply(
        lambda x: filter_sequence_by_vocab(x, vocab)
    )

    # Generate simulations
    results = []
    print(f"Worker {worker_id}: Processing {len(df_chunk)} hospitalizations...")

    for idx, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Worker {worker_id}", position=worker_id):
        input_sequence = row['filtered_clif_text']

        if not input_sequence.strip():
            print(f"Worker {worker_id}: WARNING: Empty sequence for index {idx}, skipping")
            continue

        # Generate simulations
        simulations = generate_simulations(
            model=model,
            tokenizer=tokenizer,
            input_sequence=input_sequence,
            n_simulations=n_simulations,
            max_tokens=max_tokens,
        )

        # Create result records
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

            results.append(result)

    print(f"Worker {worker_id}: Completed {len(results)} simulations")
    return results


def load_test_data(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load test datasets from parquet files."""
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


def main():
    parser = argparse.ArgumentParser(
        description='Generate clinical trajectory simulations using CPU multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model-type', type=str, required=True, choices=['gpt2_hf', 'qwen2'], help='Model type')

    # Data arguments
    parser.add_argument('--input-dir', type=str, default='benchmark/data', help='Input directory')
    parser.add_argument('--dataset', type=str, choices=['disposition', 'respiratory', 'both'], default='both')

    # Generation arguments
    parser.add_argument('--n-simulations', type=int, default=10, help='Simulations per hospitalization')
    parser.add_argument('--max-tokens', type=int, default=5000, help='Max tokens per simulation')

    # Output arguments
    parser.add_argument('--output', type=str, required=True, help='Output parquet file')

    # CPU optimization arguments
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--threads-per-worker', type=int, default=4, help='Threads per worker (default: 4)')
    parser.add_argument('--quantize', action='store_true', help='Use INT8 dynamic quantization for CPU')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process')

    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    print("=" * 80)
    print("CPU-OPTIMIZED CLINICAL TRAJECTORY SIMULATION GENERATION")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Simulations per hospitalization: {args.n_simulations}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output: {args.output}")
    print(f"CPU Workers: {args.num_workers}")
    print(f"Threads per worker: {args.threads_per_worker}")
    print()

    # Load test data
    print("Loading test data...")
    input_dir = Path(args.input_dir)
    disposition_df, respiratory_df = load_test_data(input_dir)
    print()

    # Determine datasets to process
    datasets_to_process = []
    if args.dataset in ['disposition', 'both'] and disposition_df is not None:
        datasets_to_process.append(('disposition', disposition_df))
    if args.dataset in ['respiratory', 'both'] and respiratory_df is not None:
        datasets_to_process.append(('respiratory', respiratory_df))

    if not datasets_to_process:
        print("ERROR: No datasets to process!")
        return

    # Process each dataset
    for dataset_name, df in datasets_to_process:
        print(f"\nProcessing {dataset_name} dataset...")
        print(f"  Total hospitalizations: {len(df)}")

        # Map dataset to task folder
        task_folder = 'task3-task4' if dataset_name == 'respiratory' else 'task1-task2'

        # Limit samples if specified
        if args.max_samples:
            df = df.head(args.max_samples)
            print(f"  Processing first {len(df)} hospitalizations")

        # Split data into chunks for workers
        chunk_size = len(df) // args.num_workers
        df_chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size].copy() for i in range(args.num_workers)]

        # Handle remainder
        remainder = len(df) % args.num_workers
        if remainder > 0:
            df_chunks[-1] = pd.concat([df_chunks[-1], df.iloc[-remainder:]]).reset_index(drop=True)

        print(f"  Split into {args.num_workers} chunks of ~{chunk_size} hospitalizations each")
        print(f"  Starting multiprocessing generation...")

        # Create worker function with fixed parameters
        worker_fn = partial(
            worker_process,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            n_simulations=args.n_simulations,
            max_tokens=args.max_tokens,
            num_threads=args.threads_per_worker,
            dataset_name=dataset_name,
            quantize=args.quantize,
        )

        # Run workers in parallel
        with mp.Pool(processes=args.num_workers) as pool:
            worker_args = [(i, chunk) for i, chunk in enumerate(df_chunks)]
            all_results = pool.starmap(worker_fn, worker_args)

        # Flatten results from all workers
        dataset_results = []
        for worker_results in all_results:
            dataset_results.extend(worker_results)

        print(f"\nSaving {dataset_name} results...")
        results_df = pd.DataFrame(dataset_results)

        # Add simulation ID
        results_df.insert(0, 'simulation_id', range(len(results_df)))

        print(f"  Total simulations generated: {len(results_df)}")
        print(f"  Columns: {list(results_df.columns)}")

        # Construct output path
        output_path = Path(args.output)
        base_dir = output_path.parent / 'simulations' / task_folder / args.model_type
        base_dir.mkdir(parents=True, exist_ok=True)

        final_output_path = base_dir / output_path.name
        results_df.to_parquet(final_output_path, index=False)
        print(f"  ✓ Results saved to: {final_output_path}")

    print()
    print("=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
