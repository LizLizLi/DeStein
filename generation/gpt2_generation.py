from pathlib import Path
from typing import Union, List

import torch
import torch.nn.functional as F

# from transformers import AutoAdapterModel  # 旧版本
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2PreTrainedModel,
    AutoConfig,
    GenerationConfig,
)
from transformers import GPT2LMHeadModel as GPT2_Base


from utils import utils
import numpy as np

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


class GPT2Generation:
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self,
        model: Union[str, Path, GPT2PreTrainedModel] = "gpt2",
        tokenizer: str = "",
        seed: int = 42,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, pad_token=self.STOP_TOKEN
        )
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

        # Set up model
        model_name_or_path = str(model)
        if isinstance(model, Path) or isinstance(model, str):
            if model_name_or_path.endswith(".ckpt"):
                checkpoint = torch.load(model_name_or_path)
                config = AutoConfig.from_pretrained(tokenizer)
                model_state_dict = checkpoint["state_dict"]
                # 创建一个不包含`model.`的新OrderedDict
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    name = k[6:]  # 去掉 `model.`
                    if ".attn.masked_bias" in str(name) or ".attn.bias" in str(name):
                        pass
                    else:
                        new_state_dict[name] = v
                    # new_state_dict[name] = v
                model = GPT2_Base(config=config)  # 实例化模型
                print(model)
                model.load_state_dict(new_state_dict)
            elif model_name_or_path.endswith(".pth"):
                checkpoint = torch.load(model_name_or_path, map_location="cpu")

                n_extra_tokens = 5
                tree_tokens = [
                    " _TREE_TOKEN_{}".format(str(idx).zfill(5))
                    for idx in range(n_extra_tokens)
                ] + [" _TREE_TOKEN_ZERO_COMMENTS"]
                self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

                model = GPT2_Base.from_pretrained(tokenizer)

                weights = model.get_input_embeddings().weight.detach().numpy()
                mean_weights, std_weights = np.mean(weights, axis=0), np.std(
                    weights, axis=0
                )
                new_inits = np.vstack(
                    [
                        np.random.normal(loc=mean_weights, scale=std_weights)
                        for _ in tree_tokens
                    ]
                )
                model.resize_token_embeddings(len(self.tokenizer))
                with torch.no_grad():
                    new_inits = torch.tensor(new_inits)
                    model.get_input_embeddings().weight[
                        -len(tree_tokens) :, :
                    ] = new_inits

                model.load_state_dict(checkpoint["policy_model"])
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model = model.to(self.device)
        # print(self.model)

    def __repr__(self):
        return f'<GPT2Generator model_name_or_path="{self.model}">'

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_len: int = 20,
        sample: bool = True,
        k: int = 0,
        p: float = 1.0,
        temperature: float = 1.0,
        **model_kwargs,
    ) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(
            prompt, padding=True, return_tensors="pt"
        )

        input_ids = encodings_dict["input_ids"].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_len,
                do_sample=sample,
                temperature=temperature,
                top_p=p,
                top_k=k,
                output_hidden_states=True,
            )
        decoded_outputs = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in generated_ids[:, input_seq_len:]
        ]
        return decoded_outputs
