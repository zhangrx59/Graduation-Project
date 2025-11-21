from transformers import pipeline
from PIL import Image
import requests
import torch
from huggingface_hub import logging
logging.set_verbosity_debug()

# 加载模型
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    dtype=torch.bfloat16,
    device="cuda",   # 或 device="cpu"（但推理速度慢）
)

# 准备用于输入的图像
image_path = r"C:\Users\zhangrx59\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2\HAM10000_images_part_1\ISIC_0024306.jpg"

image = Image.open(image_path).convert("RGB")
# 构造对话／提示
messages = [
    {"role":"system","content":[{"type":"text","text":"You are an expert radiologist."}]},
    {"role":"user","content":[
        {"type":"text","text":"Describe findings in this X-ray:"},
        {"type":"image","image":image}
    ]},
]

# 执行推理
output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
