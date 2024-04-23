# TODO: add `text` key to cached generations
# TODO: consolidate code for loading cache
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Iterable, List

# import openai
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm

# from transformers.pipelines import pipeline

from generation.gpt2_generation import GPT2Generation
from generation.llama2_generation import LLama2Generation
from generation.llama2_act_generation import LLama2ActGeneration
from generation.gpt2_act_generation import GPT2ActGeneration
from generation.opt_act_generation import OPTActGeneration
from generation.mpt_act_generation import MPTActGeneration
from utils.constants import OPENAI_API_KEY
from utils.utils import batchify, load_cache
from typing import Tuple, Callable, Union

logging.disable(logging.CRITICAL)  # Disable logging from transformers


def _gpt2_helper(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    generator: GPT2Generation,
    out_file: Path,
    **generate_kwargs,
):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(
        batchify(prompts, batch_size),
        total=math.ceil(len(prompts) / batch_size),
        desc=f"Generation",
        dynamic_ncols=True,
        postfix={"batch_size": batch_size},
    ):
        # Generate
        # prompt：["text1", "text2", "text3"] 未编码的一个 batch 的 prompt
        batch = generator.generate(prompt, max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open("a") as f:
                print(
                    json.dumps(generation), file=f
                )  # 将 JSON 格式的字符串输出到文件对象 f 中
            yield generation


def gpt2(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = GPT2Generation(
        model=model_name_or_path,
        tokenizer=tokenizer,
        seed=seed,
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )


def gpt2act(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    alpha,
    count,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = GPT2ActGeneration(model_name_or_path, tokenizer, seed, alpha, count)

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )


def llama2act(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    alpha,
    count,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = LLama2ActGeneration(
        model=model_name_or_path,
        tokenizer=tokenizer,
        seed=seed,
        alpha=alpha,
        count=count,
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )


def llama2(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = LLama2Generation(model_name_or_path, tokenizer, seed)

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )


def optact(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    alpha,
    count,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = OPTActGeneration(
        model=model_name_or_path,
        tokenizer=tokenizer,
        seed=seed,
        alpha=alpha,
        count=count,
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )


def mptact(
    prompts: pd.Series,
    max_len: int,
    num_samples: int,
    batch_size: int,
    model_name_or_path: str,
    tokenizer: str,
    out_file: Path,
    seed,
    alpha,
    count,
    **generate_kwargs,
) -> Iterable[str]:
    # Setup model
    generator = MPTActGeneration(
        model=model_name_or_path,
        tokenizer=tokenizer,
        seed=seed,
        alpha=alpha,
        count=count,
    )

    yield from _gpt2_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs,
    )
