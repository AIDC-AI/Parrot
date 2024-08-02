from dataclasses import dataclass, field

import torch

from parrot.model.parrot_arch import ParrotMetaForCausalLM
from parrot.utils.constants import DEFAULT_IMAGE_TOKEN
from parrot.utils.utils import disable_torch_init
from parrot.utils.mm_utils import process_images

from PIL import Image

class BaseRunner:
    def __init__(self, model_path, model_name, mm_vision_tower):
        disable_torch_init()
        self.model, self.tokenizer, self.conversation_formatter = ParrotMetaForCausalLM.build(model_name, model_path,
                                                                              mm_vision_tower=mm_vision_tower,
                                                                              low_cpu_mem_usage=True,)
                                                                            #   device_map='auto')
        self.model = self.model.cuda()
        self.image_processor = self.model.get_vision_tower().image_processor

    def truncate_text(self, text, max_length=128):
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_length:
            return text
        return self.tokenizer.decode(tokens[:max_length])

    def run(self, **kwargs):
        raise NotImplementedError


class TestRunner(BaseRunner):
    def _get_input(self, image, text):
        query = DEFAULT_IMAGE_TOKEN + '\n' + text
        prompt, input_ids = self.conversation_formatter.format_query(query)

        image = Image.open(image).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)

        return prompt, input_ids, image_tensor

    def run(self, image: Image.Image, text, **gen_args):
        prompt, input_ids, image_tensor = self._get_input(
            image=image,
            text=text
        )

        input_ids = input_ids.to(device='cuda').unsqueeze(0)
        image_tensor = image_tensor.to(dtype=self.model.dtype, device='cuda')

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                repetition_penalty=None,
                max_new_tokens=1024,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_args)

        input_token_len = input_ids.shape[1]
        output_token_len = output_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        output = output.strip()

        response = dict(
            content=output,
            prompt_tokens=input_token_len,
            total_tokens=output_token_len
        )

        print("Prompt:", prompt)
        print("Output:", output)

        return response


if __name__ == '__main__':
    model_path = '' # TODO: specify model_path
    model_name = 'parrot_qwen2'
    mm_vision_tower = '' # TODO: clip-vit-large-patch14-336

    # initialize runner
    runner = TestRunner(model_path, model_name, mm_vision_tower)
    
    # data
    image = '' # TODO: put your image path here

    texts = ["Please write a description for the image."]
    # run
    for text in texts:
        response = runner.run(image, text)
        print(response)
        print('\n')