"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    AutoConfig,
)
from modeling.modeling_opt import OPTForCausalLM
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
)


# import openai
# import httpx
# openai.api_key = ""
# openai.base_url="https://oneapi.xty.app/v1"
# openai.http_client=httpx.Client(
#             base_url="https://oneapi.xty.app/v1",
#             follow_redirects=True,
#         )

# def conditional_perplexity_openai(generations_df):
#     print("-----------start----------")
#     perplexities = []
#     # for every prompt
#     for i, row in tqdm(
#         generations_df.iterrows(),
#         total=len(generations_df.index),
#         desc="Evaluating fluency",
#     ):
#         prompt = row.prompt["text"]
#         response = openai.Completion.create(
#             engine='davinci',
#             prompt=prompt,
#             max_tokens=0,
#             temperature=0.0,
#             logprobs=0,
#             echo=True,
#         )
#         prompt_logprobs = response['choices'][0]['logprobs']['token_logprobs'][1:]
#         # for every generation conditioned on the prompt
#         generations = [g["text"] for g in row["generations"]]
#         for gen in generations:
#             generated_text = prompt+gen
#             response = openai.Completion.create(
#                 engine='davinci',
#                 prompt=generated_text,
#                 max_tokens=0,
#                 temperature=0.0,
#                 logprobs=0,
#                 echo=True,
#             )
#             logprobs = response['choices'][0]['logprobs']['token_logprobs'][1:]

#             continuation_logprobs = logprobs[len(prompt_logprobs):]
#             ppl = np.exp(-np.mean(continuation_logprobs))
#             if ppl < 1e4:  # 为保持一致性，继续使用这个sanity check
#                 perplexities.append(ppl)
#         return np.nanmean(perplexities)


def conditional_perplexity(generations_df, model, tokenizer, device="cuda"):
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc="Evaluating fluency",
    ):
        try:
            prompt = row.prompt["text"]
        except:
            prompt = row.prompt
        prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (
            prompt_input_ids.shape[1] - 1
        )
        # for every generation conditioned on the prompt
        generations = [g["text"] for g in row["generations"]]
        for gen in generations:
            full_input_ids = tokenizer.encode(prompt + gen, return_tensors="pt").to(
                device
            )
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (
                full_input_ids.shape[1] - 1
            )
            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )
            ppl = math.exp(loss.item())
            if ppl < 1e4:  # for sanity
                perplexities.append(ppl)
    return np.nanmean(perplexities)


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc="Evaluating diversity",
    ):
        generations = [g["text"] for g in row["generations"]]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(" ")
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + "_" + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + "_" + o[i + 1] + "_" + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


@click.command()
@click.option(
    "--generations_file",
    required=True,
    type=str,
    help="a jsonl file with generations and attribute scores",
)
def main(generations_file):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)

    # calculate diversity
    dist1, dist2, dist3 = distinctness(generations_df)

    # write output results
    with open(output_dir / "eval_results.txt", "w") as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f"dist-{i+1} = {dist_n}\n")

    # calculate fluency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(output_dir)
    if "llama" in str(output_dir):
        print("llama")

        config = LlamaConfig.from_pretrained("/nfs-data/user30/model/llama2-13b-hf")
        no_split_module_classes = LlamaForCausalLM._no_split_modules
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)
        device_map = infer_auto_device_map(
            model,
            max_memory={
                # 0: "5GiB",
                # 1: "5GiB",
                # 2: "20GiB",
                # 3: "20GiB",
                # 3: "10GiB",
                4: "20GiB",
                5: "20GiB",
            },
            no_split_module_classes=no_split_module_classes,
        )
        eval_model = LlamaForCausalLM.from_pretrained(
            "/nfs-data/user30/model/llama2-13b-hf",
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        # eval_model = AutoModelForCausalLM.from_pretrained(
        #     "/nfs-data/user30/model/llama2-13b-hf",
        #     torch_dtype=torch.float16,
        # ).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(
            "/nfs-data/user30/model/llama2-13b-hf"
        )
    elif "opt" in str(output_dir):
        print("opt")

        config = AutoConfig.from_pretrained("/nfs-data/user30/model/opt-6.7b")
        no_split_module_classes = OPTForCausalLM._no_split_modules
        with init_empty_weights():
            model = OPTForCausalLM._from_config(config, torch_dtype=torch.float16)
        eval_model = AutoModelForCausalLM.from_pretrained(
            "/nfs-data/user30/model/opt-6.7b",
            torch_dtype=torch.float16,
        ).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained(
            "/nfs-data/user30/model/opt-6.7b"
        )
    elif "gpt2-large" in str(output_dir):
        print("gpt")
        eval_model = AutoModelForCausalLM.from_pretrained(
            "/nfs-data/user30/model/gpt2-xl"
        ).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained("/nfs-data/user30/model/gpt2-xl")
    elif "mpt-7b" in str(output_dir):
        print("mpt")
        eval_model = AutoModelForCausalLM.from_pretrained(
            "/nfs-data/user30/model/mpt-7b",
            torch_dtype=torch.float16,
        ).to(device)
        eval_tokenizer = AutoTokenizer.from_pretrained("/nfs-data/user30/model/mpt-7b")
    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(
            generations_df, eval_model, eval_tokenizer, device=device
        )

    # write output results
    with open(output_dir / "eval_results.txt", "a") as fo:
        fo.write(f"perplexity = {ppl}")


if __name__ == "__main__":
    main()
