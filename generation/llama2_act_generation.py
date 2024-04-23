from pathlib import Path
from typing import Union, List
import pickle
import os

import torch

# from transformers import AutoAdapterModel  # 旧版本
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
)
from utils import utils
from utils.lss_utils import (
    tokenize_pairs,
    model_with_layeractadd,
    model_with_headactadd,
)
from utils.probs import get_top_heads, _merge_heads
from utils import utils
from utils.act_utils import get_act
from utils.data_utils import get_data

from modeling.modeling_llama import LlamaForCausalLM, LlamaPreTrainedModel

import re

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


class LLama2ActGeneration:
    def __init__(
        self,
        model: Union[str, Path, LlamaPreTrainedModel] = "llama2",
        tokenizer: str = "",
        seed: int = 42,
        alpha: float = 0.3,
        count: int = 20,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed=seed, n_gpu=n_gpu)

        # Set up model
        model_name_or_path = str(model)
        if isinstance(model, Path) or isinstance(model, str):
            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id
        # load activations
        parent_path = os.path.abspath(".") + "/data/act/llama2-7b/{count}/".format(
            count=count
        )
        layer_path = parent_path + "/layer.pkl"
        head_path = parent_path + "/head.pkl"
        mlp_path = parent_path + "/mlp.pkl"

        num_to_intervene = 360
        matrix_path = parent_path + "/head_matrix_{num_to_intervene}.pkl".format(
            num_to_intervene=num_to_intervene
        )
        acc_path = parent_path + "/head_acc.pkl"

        if (
            not os.path.exists(layer_path)
            or not os.path.exists(head_path)
            or not os.path.exists(mlp_path)
        ):
            data_list = get_data(count, seed)
            group_size = 20
            pairs_grouped_list = [
                data_list[i : i + group_size]
                for i in range(0, len(data_list), group_size)
            ]

            all_layer_wise_directions = []
            all_head_wise_directions = []
            all_mlp_wise_directions = []
            all_train_layer_y = []
            all_train_layer_n = []
            all_train_head_y = []
            all_train_head_n = []
            all_train_mlp_y = []
            all_train_mlp_n = []
            for index, pairs_group in enumerate(pairs_grouped_list):
                tokenize_pairs_group = tokenize_pairs(self.tokenizer, pairs_group)
                (
                    train_layer_y_all,
                    train_layer_n_all,
                    layer_wise_directions,
                    train_head_y_all,
                    train_head_n_all,
                    head_wise_directions,
                    train_mlp_y_all,
                    train_mlp_n_all,
                    mlp_wise_directions,
                ) = get_act(
                    model,
                    tokenize_pairs_group,
                )
                all_layer_wise_directions.append(layer_wise_directions)
                all_head_wise_directions.append(head_wise_directions)
                all_mlp_wise_directions.append(mlp_wise_directions)

                all_train_layer_y.append(train_layer_y_all)
                all_train_head_y.append(train_head_y_all)
                all_train_mlp_y.append(train_mlp_y_all)
                all_train_layer_n.append(train_layer_n_all)
                all_train_head_n.append(train_head_n_all)
                all_train_mlp_n.append(train_mlp_n_all)

            num_layers = model.config.num_hidden_layers
            num_heads = model.config.num_attention_heads

            all_layer_wise_directions = torch.cat(all_layer_wise_directions, dim=1)
            all_head_wise_directions = torch.cat(all_head_wise_directions, dim=1)
            all_mlp_wise_directions = torch.cat(all_mlp_wise_directions, dim=1)

            all_train_layer_y = torch.cat(all_train_layer_y, dim=1)  # [36,100,1280]
            all_train_head_y = torch.cat(all_train_head_y, dim=1)
            all_train_mlp_y = torch.cat(all_train_mlp_y, dim=1)
            all_train_layer_n = torch.cat(all_train_layer_n, dim=1)
            all_train_head_n = torch.cat(all_train_head_n, dim=1)
            all_train_mlp_n = torch.cat(all_train_mlp_n, dim=1)

            if not os.path.exists(matrix_path):
                matrix, top_heads, acc = get_top_heads(
                    all_train_head_y,
                    all_train_head_n,
                    num_layers,
                    num_heads,
                    num_to_intervene,
                    seed,
                )
                with open(matrix_path, "wb") as f:
                    pickle.dump(matrix, f)
                with open(acc_path, "wb") as f:
                    pickle.dump(acc, f)
            else:
                matrix = pickle.load(
                    open(
                        matrix_path,
                        "rb",
                    )
                )
                acc = pickle.load(
                    open(
                        acc_path,
                        "rb",
                    )
                )

            all_layer_wise_directions = all_layer_wise_directions.sum(
                dim=1, keepdim=False
            ).view(num_layers, -1)
            all_head_wise_directions = all_head_wise_directions.sum(
                dim=1, keepdim=False
            ).view(num_layers, -1)

            all_mlp_wise_directions = all_mlp_wise_directions.sum(
                dim=1, keepdim=False
            ).view(num_layers, -1)

            directions_paths = [
                (all_layer_wise_directions, layer_path),
                (all_head_wise_directions, head_path),
                (all_mlp_wise_directions, mlp_path),
            ]
            for direction, path in directions_paths:
                with open(path, "wb") as f:
                    pickle.dump(direction, f)
        else:
            all_layer_wise_directions = pickle.load(
                open(
                    layer_path,
                    "rb",
                )
            )
            all_head_wise_directions = pickle.load(
                open(
                    head_path,
                    "rb",
                )
            )
            all_mlp_wise_directions = pickle.load(
                open(
                    mlp_path,
                    "rb",
                )
            )
            matrix = pickle.load(
                open(
                    matrix_path,
                    "rb",
                )
            )
            acc = pickle.load(
                open(
                    acc_path,
                    "rb",
                )
            )
        acc = torch.from_numpy(acc)
        num_heads = model.config.num_attention_heads
        expanded_acc = (
            acc.unsqueeze(2)
            .expand(-1, -1, int(all_head_wise_directions.size(1) / num_heads))
            .unsqueeze(2)
        )
        reshaped_acc = _merge_heads(
            expanded_acc, int(expanded_acc.size(1)), int(expanded_acc.size(3))
        ).squeeze(1)
        mask = reshaped_acc + 1
        all_head_wise_directions = torch.mul(all_head_wise_directions, mask)

        # self.model = model_with_mlpactadd(model).get_model(
        #     torch.stack([all_layer_wise_directions], dim=1).cuda(), alpha=alpha
        # )
        self.model = model_with_headactadd(model).get_model(
            torch.stack([all_head_wise_directions], dim=1).cuda(), alpha=alpha
        )
        # self.model = model_with_layeractadd(model).get_model(
        #     torch.stack([all_head_wise_directions], dim=1).cuda(), alpha=alpha
        # )

    def __repr__(self):
        return f'<LLama2Generator model_name_or_path="{self.model}">'

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

        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        batch_size, input_seq_len = inputs["input_ids"].shape

        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_len,
                do_sample=sample,
                temperature=temperature,
                top_p=p,
                top_k=k,
            )
        decoded_outputs = [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in generated_ids[:, input_seq_len:]
        ]
        return decoded_outputs
