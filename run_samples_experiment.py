import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm
import os

# from generation.generation import gpt2, gpt3, pplm, dexperts, dexperts_gpt3
from generation.generation import gpt2, llama2, optact, mptact
from utils.utils import ensure_dir

ALLOWED_MODELS = ["gpt2", "llama2", "opt", "mpt"]


def str2bool(str):
    return True if str.lower() == "true" else False


@click.command()
@click.argument("output-dir")
@click.option(
    "--dataset-file",
    required=False,
    type=str,
    help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.',
)
@click.option(
    "--use-eos/--use-dataset",
    default=False,
    help="Whether to use EOS or a dataset file for generation.",
)
@click.option(
    "--model", required=True, help="Equivalent to `model_name_or_path` in transformers."
)
@click.option("--is-adapter", type=str2bool, default="False", help="是否加载 adapter.")
@click.option("--model-type", required=True, type=click.Choice(ALLOWED_MODELS))
@click.option(
    "--n",
    default=25,
    help="Number of samples to generate for each prompt. When used with --eos",
)
@click.option("--seed", default=42, type=int)
@click.option(
    "--max-tokens",
    default=20,
    help="Number of tokens (usually BPE) to generate for each prompt.",
)
@click.option("--batch-size", default=32)
@click.option("--resume/--no-resume", default=False)
@click.option(
    "--p", default=1.0, type=float, help="Hyperparameter for nucleus sampling"
)
def main(
    output_dir: str,
    dataset_file: Optional[str],
    use_eos: bool,
    model: str,
    is_adapter: bool,
    model_type: str,
    n: int,
    max_tokens: int,
    batch_size: int,
    resume: bool,
    p: float,
    seed: int,
):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith(".jsonl")
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset["prompt"])["text"]
    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ["gpt2"]:
            prompts = pd.Series("<|endoftext|>")
        elif model_type in ["llama2"]:
            prompts = pd.Series("")
        else:
            raise RuntimeError("Model not implemented with EOS prompts")
    else:
        raise click.exceptions.MissingParameter(
            "Missing --dataset-file or --use-eos option."
        )
    print("Prompts:", "\n", prompts)

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / "generations.jsonl"  # 生成续文文件
    assert resume or not os.path.exists(
        generations_file
    )  # don't overwrite generations!
    ensure_dir(output_dir)

    # Setup model for generation
    if model_type == "gpt2":
        generations_iter = gpt2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            is_adapter=is_adapter,
            out_file=generations_file,
            seed=seed,
        )
    elif model_type == "llama2":
        generations_iter = llama2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=model,
            out_file=generations_file,
            seed=seed,
        )
    elif model_type == "opt":
        generations_iter = optact(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=model,
            out_file=generations_file,
            seed=seed,
        )
    elif model_type == "mpt":
        generations_iter = mptact(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=model,
            out_file=generations_file,
            seed=seed,
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented")

    # Generate and collate perspective scores
    generations = []
    for gen in enumerate(generations_iter):
        generations.append(gen)
    torch.cuda.empty_cache()
    print("Finished generation!")


if __name__ == "__main__":
    main()
