import json

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import base64
import io
import os
from transformers import AutoTokenizer, AutoModel
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class TPOChat:
    def __init__(self, model_path) -> None:
        disable_torch_init()
        model_name='llava-v1.5-7b'
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None,model_name=model_name, device_map={"": 'cuda'})
        self.tokenizer=tokenizer
        self.model=model
        self.image_processor=image_processor
        self.context_len=context_len

    def chat(self, input):
        msgs = input['question']
        if self.model.config.mm_use_im_start_end:
            msgs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + msgs
        else:
            msgs = DEFAULT_IMAGE_TOKEN + '\n' + msgs

        image = Image.open(input['image']).convert('RGB')
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], msgs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                num_beams=3,
                max_new_tokens=1024,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


if __name__ == '__main__':
    chat_model = TPOChat('checkpoint/tpo-llava15_7b_main/checkpoints')
    while True:
        print("image_paths:")
        image_path=input() # examples/example.jpeg
        print("question:")
        msgs =input() # "Why did the car in the picture stop?"
        inputs = {"image": image_path, "question": msgs}
        answer = chat_model.chat(inputs)
        print(answer)
