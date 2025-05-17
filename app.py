from flask import Flask, request, jsonify, render_template
from PIL import Image
import os
import torch
from inference import load_model_components, run_inference

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./localmodel"

# 模型组件加载（只初始化一次）
print("📦 加载模型中...")
model, tokenizer, processor = load_model_components(model_path, device)
print("✅ 模型加载完成")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    save_path = os.path.join("tmp_uploads", file.filename)
    os.makedirs("tmp_uploads", exist_ok=True)
    file.save(save_path)

    try:
        image = Image.open(save_path)
        latex = run_inference(image, model, tokenizer, processor, device)
        return jsonify({"latex": latex or "未识别出公式"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
