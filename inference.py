# inference.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "nougat-latex-ocr"))

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex import NougatLaTexProcessor

# 初始化一次（可复用）
def load_model_components(model_path, device):
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    tokenizer = NougatTokenizerFast.from_pretrained(model_path)
    processor = NougatLaTexProcessor.from_pretrained(model_path)
    return model, tokenizer, processor

# 核心推理函数：输入图像对象，输出 LaTeX 字符串
def run_inference(image: Image.Image, model, tokenizer, processor, device):
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = tokenizer(
        tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=5,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    return sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "").strip()
