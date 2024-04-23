import torch
from torch.nn import functional as F


class ActLayer(torch.nn.Module):
    def __init__(self, config, acts, alpha, n):
        super(ActLayer, self).__init__()
        self.config = config
        self.acts = acts
        self.alpha = alpha
        self.n = n
        self.weight_all = []

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, x, is_multihead=False):
        input_dtype = x.dtype
        n = self.n + 1
        norm = torch.norm(x.float(), dim=-1).unsqueeze(-1)
        alpha = self.alpha

        if self.acts is not None:
            if is_multihead:
                x_multihead = self._split_heads(
                    x.float(),
                    self.config.num_attention_heads,
                    int(x.size(2) / self.config.num_attention_heads),
                ).permute(
                    0, 2, 1, 3
                )  # (batch, seq_length, head, head_features)
                acts_multihead = self._split_heads(
                    self.acts.unsqueeze(0),
                    self.config.num_attention_heads,
                    int(self.acts.size(-1) / self.config.num_attention_heads),
                ).permute(
                    0, 2, 1, 3
                )  # [1,1,20,64]
                cos_sim = F.cosine_similarity(x_multihead, acts_multihead, dim=-1)
                lambda_sim = 1.0 + torch.max(
                    torch.tensor([0.0], device=x.device), cos_sim
                ).unsqueeze(
                    -1
                )  # 只有第一次是 torch.Size([25, 12, 20, 1]) 后面都是 [25, 1, 20, 1]
                acts_normalized = F.normalize(
                    acts_multihead, dim=-1
                )  # torch.Size([1, 1, 20, 64])
                acts = alpha * lambda_sim * acts_normalized.repeat(1, x.shape[1], 1, 1)
                # acts = alpha * acts_normalized.repeat(1, x.shape[1], 1, 1)
                x_updated = F.normalize(
                    F.normalize(x_multihead.float(), dim=-1) - acts,
                    dim=-1,
                ).permute(
                    0, 2, 1, 3
                )  # 更新 x 并再次归一化
                # 调整 x 的大小
                x_new = self._merge_heads(
                    x_updated.float(),
                    self.config.num_attention_heads,
                    int(x.size(2) / self.config.num_attention_heads),
                )

                new_norm = x_new.norm(dim=-1).unsqueeze(-1)
                x = x_new * (norm / new_norm)
                return x.type(input_dtype)
            else:
                act = alpha * F.normalize(self.acts[0], dim=-1).repeat(1, x.shape[1], 1)
                x = F.normalize(F.normalize(x.float(), dim=-1) - act, dim=-1)
                new_norm = x.norm(dim=-1).unsqueeze(-1)
                x = x * (norm / new_norm)
                return x.type(input_dtype)
        else:
            return x


class model_with_actadd(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, acts, alpha):
        for i in range(0, len(self.model.transformer.h)):
            acts_ = acts[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(
                self.model.transformer.h[i].mlp, ActLayer(acts_, alpha)
            )
        return self.model


class model_with_layeractadd(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, acts, alpha):
        config = self.model.config
        model_name = config.model_type
        if "gpt2" in model_name:
            for i in range(0, len(self.model.transformer.h)):
                acts_ = acts[i]
                self.model.transformer.h[i].layer_out = torch.nn.Sequential(
                    self.model.transformer.h[i].layer_out, ActLayer(acts_, alpha)
                )
        elif "llama2" in model_name:
            for i in range(0, len(self.model.model.layers)):
                acts_ = acts[i]
                self.model.model.layers[i].layer_out = torch.nn.Sequential(
                    self.model.model.layers[i].layer_out,
                    ActLayer(acts_, alpha),
                )
        return self.model


class model_with_headactadd(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, acts, alpha):
        config = self.model.config
        model_name = config.model_type
        if "gpt2" in model_name:
            for i in range(0, len(self.model.transformer.h)):
                acts_ = acts[i]
                self.model.transformer.h[i].attn.head_out = torch.nn.Sequential(
                    self.model.transformer.h[i].attn.head_out,
                    ActLayer(config, acts_, alpha, i),
                )  # .head_out
        elif "llama" in model_name:
            for i in range(0, len(self.model.model.layers)):
                acts_ = acts[i]
                self.model.model.layers[i].self_attn.head_out = torch.nn.Sequential(
                    self.model.model.layers[i].self_attn.head_out,
                    ActLayer(config, acts_, alpha, i),
                )
        elif "opt" in model_name:
            for i in range(0, len(self.model.model.decoder.layers)):
                acts_ = acts[i]
                self.model.model.decoder.layers[i].self_attn.head_out = (
                    torch.nn.Sequential(
                        self.model.model.decoder.layers[i].self_attn.head_out,
                        ActLayer(config, acts_, alpha, i),
                    )
                )
        elif "mpt" in model_name:
            for i in range(0, len(self.model.transformer.blocks)):
                acts_ = acts[i]
                self.model.transformer.blocks[i].attn.head_out = torch.nn.Sequential(
                    self.model.transformer.blocks[i].attn.head_out,
                    ActLayer(config, acts_, alpha, i),
                )
        return self.model


class model_with_mlpactadd(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, acts, alpha):
        config = self.model.config
        model_name = config.model_type
        if "gpt2" in model_name:
            for i in range(0, len(self.model.transformer.h)):
                acts_ = acts[i]
                self.model.transformer.h[i].mlp = torch.nn.Sequential(
                    self.model.transformer.h[i].mlp, ActLayer(acts_, alpha)
                )
        elif "llama2" in model_name:
            for i in range(0, len(self.model.model.layers)):
                acts_ = acts[i]
                self.model.model.layers[i].mlp = torch.nn.Sequential(
                    self.model.model.layers[i].mlp,
                    ActLayer(acts_, alpha),
                )
        return self.model


def tokenize_pairs(tokenizer, pairs):
    tokenize_pairs = []
    for i in range(len(pairs)):
        pairs[i] = (
            pairs[i][0].strip(" .").strip("."),
            pairs[i][1].strip(" .").strip("."),
        )

        tox = tokenizer(pairs[i][0])
        notox = tokenizer(pairs[i][1])
        tokenize_pairs.append((tox, notox))
    return tokenize_pairs
