import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import ast
import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
from typing import Tuple, Callable, Union

import click
import pandas as pd
import torch
from tqdm import tqdm
import os

from generation.generation import (
    gpt2,
    llama2,
    llama2act,
    gpt2act,
    optact,
    mptact,
)
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.utils import load_jsonl, batchify, ensure_dir

ALLOWED_MODELS = [
    "gpt2",
    "llama2",
    "llama2-act",
    "gpt2-act",
    "opt-act",
    "mpt-act",
]


def str2bool(str):
    return True if str.lower() == "true" else False


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        if response["response"]:
            response = unpack_scores(response["response"])[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {"text": generation, **response}


def collate(
    dataset: Optional[pd.DataFrame],
    generations: List[str],
    responses: Iterable[Dict[str, Any]],
    output_file: str,
):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(
            tqdm(generations_col_iter, total=len(generations), desc="Collating files")
        )
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(
            tqdm(
                batchify(generations_col_iter, n),
                total=len(dataset),
                desc="Collating files",
            )
        )
        dataset["generations"] = generations_col

    dataset.to_json(output_file, orient="records", lines=True)


@click.command()
@click.argument(
    "output-dir",
    default="/nfs-data/user30/Projects/00MY/00DeStein/generations/results/gpt2-large/test",
)
@click.option(
    "--dataset-file",
    required=False,
    type=str,
    default="/nfs-data/user30/Projects/00MY/00DeStein/data/RealToxicityPrompts/100/test.jsonl",
    help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.',
)
@click.option(
    "--model",
    required=True,
    default="/nfs-data/user30/model/gpt2-large",
    help="Equivalent to `model_name_or_path` in transformers.",
)
@click.option(
    "--tokenizer",
    default="/nfs-data/user30/model/gpt2-large",
    type=str,
    help="Equivalent to `tokenizer_name_or_path` in transformers.",
    callback=lambda c, p, v: v if v else c.params["model"],
)
@click.option(
    "--model-type",
    required=True,
    type=click.Choice(ALLOWED_MODELS),
    default="gpt2-act",
)
@click.option("--perspective-rate-limit", default=30)
@click.option(
    "--n",
    default=25,
    help="Number of samples to generate for each prompt. When used with --eos",
)
@click.option(
    "--max-tokens",
    default=20,
    help="Number of tokens (usually BPE) to generate for each prompt.",
)
@click.option("--batch-size", default=25)
@click.option("--resume/--no-resume", default=False)
@click.option(
    "--p", default=0.9, type=float, help="Hyperparameter for nucleus sampling"
)
@click.option("--seed", default=16, type=int)
@click.option("--alpha", default=0.4, type=float)
@click.option("--count", default=20, type=int)
def main(
    output_dir: str,
    dataset_file: Optional[str],
    model: str,
    tokenizer: str,
    model_type: str,
    perspective_rate_limit: int,
    n: int,
    max_tokens: int,
    batch_size: int,
    resume: bool,
    p: float,
    seed: int,
    alpha: float,
    count: int,
):
    # Load prompts
    if dataset_file:
        # Load prompts from dataset file
        assert dataset_file.endswith(".jsonl")
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset["prompt"])["text"]
        if model_type == "lss":
            prompts = [
                "The following text is kind, polite, positive, respectful and supportive: "
                + prompt
                for prompt in prompts
            ]
            prompts = pd.Series(prompts)
    else:
        raise click.exceptions.MissingParameter("Missing --dataset-file.")
    print("Prompts:", "\n", prompts)

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / "generations.jsonl"  # 生成续文文件
    perspective_file = output_dir / "perspective.jsonl"  # 评分文件
    assert resume or not os.path.exists(
        generations_file
    )  # don't overwrite generations!
    ensure_dir(output_dir)
    output_file = output_dir / f"prompted_gens_{model_type}.jsonl"

    # Create perspective worker thread
    perspective = PerspectiveWorker(
        out_file=perspective_file,
        total=len(prompts) * n,
        rate_limit=perspective_rate_limit,
    )

    # Setup model for generation
    if model_type == "gpt2":
        generations_iter = gpt2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
        )
    elif model_type == "gpt2-act":
        generations_iter = gpt2act(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    elif model_type == "llama2-act":
        generations_iter = llama2act(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    elif model_type == "llama2":
        generations_iter = llama2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    elif model_type == "opt-act":
        generations_iter = optact(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    elif model_type == "mpt-act":
        generations_iter = mptact(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            tokenizer=tokenizer,
            out_file=generations_file,
            seed=seed,
            alpha=alpha,
            count=count,
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented")

    T1 = time.time()

    # Generate and collate perspective scores
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)
        perspective(f"generation-{i}", gen)

    T2 = time.time()
    print("程序运行时间:%s毫秒" % ((T2 - T1) * 1000))

    torch.cuda.empty_cache()
    perspective.stop()
    print("Finished generation and perspective scoring!")

    if os.path.exists(perspective_file):
        print("Collating output files")
        collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == "__main__":
    main()
