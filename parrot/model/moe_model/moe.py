import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN


class Expert(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.moe_hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.moe_hidden_size
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, config.num_experts),
        )

    def forward(self, inputs):
        return torch.softmax(self.gate(inputs), dim=-1)


class MultilingualMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.moe_hidden_size
        self.num_experts = config.num_experts
        self.gate = nn.Linear(self.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )

        self.use_moe_residual = getattr(config, 'use_moe_residual', True)
        self.use_moe = getattr(config, 'use_moe', True)
        self.moe_top_k = getattr(config, 'moe_top_k', self.experts)
        self.moe_weight = getattr(config, 'moe_weight', 1.0)

    def tune_model(self, config):
        # reset the param using config
        self.use_moe = config.use_moe
        self.moe_weight = config.moe_weight
        self.moe_top_k = config.moe_top_k
        self.use_moe_residual = config.use_moe_residual

        if self.use_moe:
            for name, param in self.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                param.requires_grad = False


    def forward(self, input_embeds, image_features):
        # for a single image and a single text
        if not self.use_moe:
            return image_features

        assert self.config.mm_vision_select_feature == 'cls_patch'
        cls_token = image_features[0:1, :]  # 1, hidden
        sequence_length, hidden_dim = image_features.shape

        # cross attention for image and text
        scores = torch.matmul(cls_token, input_embeds.transpose(-2, -1)) / torch.sqrt(torch.tensor(cls_token.size(-1)))
        attention_weight = F.softmax(scores, dim=-1)
        embeds_for_gating = torch.matmul(attention_weight, input_embeds)

        router_logits = self.gate(embeds_for_gating)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=image_features.dtype)  # 1, expert

        experts_output = torch.stack([expert(image_features) for expert in self.experts], dim=0)

        final_hidden_states = torch.sum(routing_weights[0].unsqueeze(-1).unsqueeze(-1) * experts_output, dim=0)

        if self.use_moe_residual:
            final_hidden_states += self.moe_weight * image_features

        return final_hidden_states
