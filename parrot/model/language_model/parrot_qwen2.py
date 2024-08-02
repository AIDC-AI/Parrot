#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..parrot_arch import ParrotMetaModel, ParrotMetaForCausalLM


class ParrotQwen2Config(Qwen2Config):
    model_type = "parrot_qwen2"


class ParrotQwen2Model(ParrotMetaModel, Qwen2Model):
    config_class = ParrotQwen2Config

    def __init__(self, config: Qwen2Config):
        super(ParrotQwen2Model, self).__init__(config)


class ParrotQwen2ForCausalLM(Qwen2ForCausalLM, ParrotMetaForCausalLM):
    config_class = ParrotQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ParrotQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.router_aux_loss_coef = 0.001  # defaults to 0.001
        self.output_router_logits = True if self.router_aux_loss_coef != 0 else False

        # Initialize weights and apply final processing
        self.post_init()

    def get_monitor_tensors(self):
        moe_bottom = self.model.moe_model.experts[0].down_proj.weight
        moe_top = self.model.moe_model.experts[-1].up_proj.weight

        monitor_tensors = dict(
                vision_tower_bottom=self.model.vision_tower.vision_tower.vision_model.encoder.layers[0].self_attn.k_proj.weight,
                vision_tower_top=self.model.vision_tower.vision_tower.vision_model.encoder.layers[-1].self_attn.out_proj.weight,
                llm_bottom=self.model.layers[0].mlp.down_proj.weight,
                llm_top=self.model.layers[-1].mlp.down_proj.weight,
                projector=self.model.mm_projector[-1].weight,
                moe_bottom=moe_bottom,
                moe_top=moe_top,
                moe_gate=self.model.moe_model.gate.weight
            )
        return monitor_tensors

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # Copied from transformers/models/qwen2/modeling_qwen2.Qwen2ForCausalLM.forward
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            attention_mask = attention_mask[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    @classmethod
    def build(cls, model_name, model_path, **kwargs):
        model_kwargs = {k: v for k, v in kwargs.items() if k in cls.MODEL_BUILD_KEYS}
        model = cls.from_pretrained(
            model_path,
            model_type=ParrotQwen2Config.model_type,
            **model_kwargs
        )

        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k in cls.TOKENIZER_BUILD_KEYS}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            eos_token="<|im_end|>",
            **tokenizer_kwargs
        )
        return model, tokenizer


AutoConfig.register("parrot_qwen2", ParrotQwen2Config)
AutoModelForCausalLM.register(ParrotQwen2Config, ParrotQwen2ForCausalLM)
