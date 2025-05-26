import re
import torch
from transformers import AutoTokenizer
from difflib import SequenceMatcher  # 用于字符级匹配

# TokenAccMetric 类
class TokenAccMetric:
    def __init__(self, pad_token_id=0, eos_token_id=2, **kwargs):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.total_tokens = 0
        self.token_correct = 0
        self.token_acc = []

    def add(self, tgt_seqs, preds):
        shape_diff = preds.shape[1] - tgt_seqs.shape[1]
        if shape_diff < 0:
            preds = torch.nn.functional.pad(preds, (0, -shape_diff), "constant", self.pad_token_id)
        elif shape_diff > 0:
            tgt_seqs = torch.nn.functional.pad(tgt_seqs, (0, shape_diff), "constant", self.pad_token_id)
        mask = torch.logical_or(tgt_seqs != self.pad_token_id, preds != self.pad_token_id)
        tok_acc = (preds == tgt_seqs)[mask].float()
        self.token_acc.append(tok_acc.mean().item())
        self.token_correct += int(tok_acc.sum().item())
        self.total_tokens += len(tok_acc)

    def mean(self):
        return self.token_correct / self.total_tokens if self.total_tokens > 0 else 0

# 设置路径
txt_file = "/root/autodl-tmp/Qwen2-VL-finetune-LatexOCR-main/output/Qwen2-VL-2B-LatexOCR-22w/predictions_baseline.txt"
local_model_path = "/root/autodl-tmp/Qwen2-VL-finetune-LatexOCR-main/Qwen/Qwen2-VL-2B-Instruct"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

# 提取预测和GT
predictions = []
ground_truths = []
with open(txt_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].strip()
    if line.startswith("Prediction:"):
        pred = line.replace("Prediction:", "").strip()
        gt = lines[i + 1].strip().replace("Ground Truth:", "").strip()
        predictions.append(pred)
        ground_truths.append(gt)

# 初始化 Token Accuracy
metric = TokenAccMetric(pad_token_id=tokenizer.pad_token_id or 0)
for pred, gt in zip(predictions, ground_truths):
    pred_ids = tokenizer(pred, return_tensors="pt", add_special_tokens=False).input_ids
    gt_ids = tokenizer(gt, return_tensors="pt", add_special_tokens=False).input_ids
    metric.add(gt_ids, pred_ids)

print(f"Token-level Accuracy: {metric.mean():.4f}")

# ===== 计算字符级准确率 =====
char_correct = 0
char_total = 0

for pred, gt in zip(predictions, ground_truths):
    matcher = SequenceMatcher(None, pred, gt)
    match_size = sum(block.size for block in matcher.get_matching_blocks())
    char_correct += match_size
    char_total += len(gt)

char_acc = char_correct / char_total if char_total > 0 else 0
print(f"Character-level Accuracy: {char_acc:.4f}")
print(f"Correct tokens: {metric.token_correct}")
print(f"Total tokens: {metric.total_tokens}")
print(f"Token Acc: {metric.mean():.4f}")

