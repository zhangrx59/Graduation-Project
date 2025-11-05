from openai import OpenAI
import base64
import requests

def encode_image_to_base64(image_path):
    """将本地图像转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


client = OpenAI(api_key="sk-uhobwdodfqwkayapmgposlzsykswpuzaunzebblhcmgyrage",
                base_url="https://api.siliconflow.cn/v1")

# 将本地图像转换为base64
image_path = "C:/Users/zhangrx59/Desktop/a.jpg"  # 使用正斜杠
base64_image = encode_image_to_base64(image_path)


response = client.chat.completions.create(
    # model='Pro/deepseek-ai/DeepSeek-R1',
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)