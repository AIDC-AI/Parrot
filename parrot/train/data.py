import logging
import os
import copy
import random
from dataclasses import dataclass

from typing import Dict, Sequence
import io
import torch

import transformers

from parrot.model.conversation_formatter import ConversationFormatter
from parrot.utils.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, BEGIN_LINE, END_LINE
from torch.utils.data import Dataset

from parrot.utils.utils import rank0_print
from parrot.utils.utils import import_class_from_string, name2data

from PIL import Image
from datetime import datetime
from arguments import DataArguments


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args: DataArguments,
                 tokenizer: transformers.PreTrainedTokenizer,
                 conversation_formatter: ConversationFormatter):
        super(SupervisedDataset, self).__init__()
        self.conversation_formatter = conversation_formatter

        rank0_print(f"[{datetime.now()}] Formatting inputs begin")
        list_data_dict = []
        for name in data_args.data_name.split('|'):
            dataset = name2data(name)
            if data_args.data_processor:
                dataset = import_class_from_string(data_args.data_processor).process_train(dataset,
                                                                                           mode=data_args.data_mode,
                                                                                           cache_dir=data_args.processed_data_dir,
                                                                                           data_name=name)
            list_data_dict.extend(dataset)
        rng = random.Random(42)
        for i in range(10):
            rng.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        rank0_print(f"[{datetime.now()}] Formatting inputs end")


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[i]
        conversations = copy.deepcopy(sample["conversations"])

        has_image = 'image' in sample
        if has_image:
            image_path = sample['image']
            processor = self.data_args.image_processor
            default_image = Image.new('RGB', (processor.crop_size['height'], processor.crop_size['width']),
                                      color=tuple(int(x * 255) for x in processor.image_mean))  # a mean image

            # read image
            image_dir = self.data_args.image_dir
            image_full_path = os.path.join(image_dir, image_path)
            try:
                image = Image.open(image_full_path).convert('RGB')
            except Exception as e:
                logging.warning(BEGIN_LINE)
                logging.warning(f'read image from {image_full_path} fail')
                logging.warning(f'exception is: {e}')
                logging.warning(END_LINE)
                image = default_image  # a mean image

            # image process
            try:
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except Exception as e:
                logging.warning(BEGIN_LINE)
                logging.warning(f'process image of {image_path} fail')
                logging.warning(f'exception is: {e}')
                logging.warning(END_LINE)
                image = default_image  # a mean image
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # process

            # align image token position
            for conversation in conversations:
                if DEFAULT_IMAGE_TOKEN in conversation['value']:
                    conversation['value'] = conversation['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    conversation['value'] = DEFAULT_IMAGE_TOKEN + '\n' + conversation['value']
                    conversation['value'] = conversation['value'].strip()

        # Now, sources are singleton list, with the element being list of dicts with two keys: `from` and `value`
        prompt, input_ids, labels = self.conversation_formatter.format(conversations)
        if not has_image and self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            image = torch.zeros(3, crop_size['height'], crop_size['width'])

        return dict(
            input_ids=input_ids,
            labels=labels,
            image=image
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(data_args: DataArguments,
                                tokenizer: transformers.PreTrainedTokenizer,
                                conversation_formatter: ConversationFormatter) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args=data_args,
                                      tokenizer=tokenizer,
                                      conversation_formatter=conversation_formatter)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
