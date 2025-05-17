import os
import json
import torch
from modelscope import snapshot_download, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =================== 设置路径 ====================
prompt = "你是一个LaText OCR助手,目标是读取用户输入的照片，转换成LaTex公式。"
model_id = "Qwen/Qwen2-VL-2B-Instruct"
local_model_path = "/root/autodl-tmp/Qwen2-VL-finetune-LatexOCR-main/Qwen/Qwen2-VL-2B-Instruct"
val_dataset_json_path = "/root/autodl-tmp/project/nougat-latex-ocr/latex_ocr_val_new.json"
output_dir = "./output/Qwen2-VL-2B-LatexOCR-22w"
MAX_LENGTH = 8192

# ============ 加载模型与分词器 ============
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)

origin_model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
origin_model.eval()

# ============ 推理函数 ============
def predict(messages, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    return output_text[0]

# ============ 读取验证集 ============
with open(val_dataset_json_path, "r", encoding="utf-8") as f:
    test_dataset = json.load(f)

# ============ 推理并写入输出 ============
output_txt_path = os.path.join(output_dir, "predictions_baseline.txt")
os.makedirs(output_dir, exist_ok=True)

with open(output_txt_path, "w", encoding="utf-8") as out_file:
    for item in test_dataset:
        image_file_path = item["conversations"][0]["value"]
        label = item["conversations"][1]["value"]

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file_path,
                    "resized_height": 100,
                    "resized_width": 500,
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]}]

        response = predict(messages, origin_model)

        print(f"predict: {response}")
        print(f"gt: {label}\n")

        out_file.write(f"Image: {image_file_path}\n")
        out_file.write(f"Prediction: {response}\n")
        out_file.write(f"Ground Truth: {label}\n")
        out_file.write("\n" + "-"*80 + "\n")

print(f"原始模型的预测结果已保存至：{output_txt_path}")
