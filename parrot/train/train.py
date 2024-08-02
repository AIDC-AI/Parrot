# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import json
import os.path
import pathlib
import torch

import transformers

from parrot.utils.constants import BEGIN_LINE, END_LINE
from parrot.model.parrot_arch import ParrotMetaForCausalLM
from parrot.utils.utils import rank0_print

from data import make_supervised_data_module
from saver import safe_save_model_for_hf_trainer
from parrot_trainer import ParrotTrainer
from arguments import ModelArguments, DataArguments, TrainingArguments


def train():
    # parse args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with training_args.main_process_first(local=False):
        if int(os.environ['RANK']) == 0:
            def args2dict(args):
                return {k: str(v) for k, v in args.__dict__.items()}

            args_log = json.dumps(dict(
                model_args=args2dict(model_args),
                data_args=args2dict(data_args),
                training_args=args2dict(training_args)
            ), ensure_ascii=False, indent=2)
            print(args_log)
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(os.path.join(training_args.output_dir, 'model_data_training_args.json'), 'w',
                      encoding='utf-8') as f:
                f.write(args_log + '\n')

    training_args.unfreeze()

    # load model
    model, tokenizer, conversation_formatter = ParrotMetaForCausalLM.build(model_args.model_name, model_args.model_path,
                                                                          model_max_length=training_args.model_max_length)
    model = model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.get_model().requires_grad_(False)

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    if model_args.freeze_vision_tower:
        vision_tower.requires_grad_(False)
    else:
        vision_tower.requires_grad_(True)

    model_args.moe_hidden_size = model.config.hidden_size  # LLM's hidden size

    model.get_model().initialize_moe_modules(
        model_args=model_args
    )

    moe_model = model.get_moe_model()
    moe_model.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    # tune MoE module in stage 2
    moe_model.tune_model(model_args)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.config.mm_projector_lr = training_args.mm_projector_lr

    data_module = make_supervised_data_module(data_args=data_args,
                                        tokenizer=tokenizer,
                                        conversation_formatter=conversation_formatter)

    # print param
    for name, param in model.get_model().named_parameters():
        if param.requires_grad:
            rank0_print(f'{name} {torch.numel(param)} is trainable')

    # train
    training_args.freeze()
    trainer = ParrotTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample tensor:')
    rank0_print(data_module['train_dataset'][0])
    rank0_print(END_LINE)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample input_ids decoding:')
    rank0_print(tokenizer.decode([x for x in data_module['train_dataset'][0]['input_ids'] if x >= 0]))
    rank0_print(END_LINE)
    rank0_print(BEGIN_LINE)
    rank0_print('Dataset sample labels decoding:')
    rank0_print(tokenizer.decode([x for x in data_module['train_dataset'][0]['labels'] if x >= 0]))
    rank0_print(END_LINE)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # save model
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
