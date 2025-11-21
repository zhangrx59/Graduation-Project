# -*- coding: utf-8 -*-
import os
import base64
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ===================== 通用工具 =====================

def load_config(config_path="config.json"):
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_image_to_base64(image_path: Path) -> str:
    """将本地图像转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_imageid_to_path(folder_path: str, supported_formats):
    """
    扫描图片文件夹，构建 {image_id(不带扩展名) -> 完整路径} 映射
    只要文件名是 XXX.jpg / XXX.png，就对应 image_id = XXX
    """
    folder = Path(folder_path)
    mapping = {}

    if not folder.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {folder}")

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in supported_formats:
            image_id = f.stem  # 去掉扩展名
            mapping[image_id] = f

    print(f"📁 在 {folder_path} 中共找到 {len(mapping)} 个 image_id 对应的图片文件")
    return mapping


def safe_read_csv(csv_file_path: str) -> pd.DataFrame:
    """
    尝试多种编码读取 CSV，解决 UnicodeDecodeError 问题
    """
    encodings_to_try = ["utf-8", "gbk", "gb2312", "latin1"]

    last_err = None
    for enc in encodings_to_try:
        try:
            print(f"尝试使用编码 {enc} 读取 CSV...")
            df = pd.read_csv(csv_file_path, encoding=enc, low_memory=False)
            print(f"✅ 使用编码 {enc} 读取 CSV 成功")
            return df
        except UnicodeDecodeError as e:
            print(f"❌ 使用编码 {enc} 失败: {e}")
            last_err = e

    # 如果都失败，则抛出最后一次错误
    raise last_err


def call_qwen_shape(client: OpenAI, model_type: str, image_path: Path, shape_prompt: str) -> str:
    """
    调用 Qwen-VL，返回“病变形状”的简短中文短语
    """
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": shape_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model_type,
        messages=messages,
        stream=False,
    )

    # SiliconFlow + OpenAI SDK 返回的一般结构
    content = resp.choices[0].message.content

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        result = "".join(texts).strip()
    else:
        result = str(content).strip()

    # 只保留第一行，去掉多余空白
    result = result.splitlines()[0].strip()
    return result


# ===================== 主流程：只处理“有图片存在”的行 =====================

def annotate_image_shapes(config_path="config.json",
                          output_csv_path=None,
                          save_every_n=20):
    """
    读取 CSV，根据 图片ID 调用 Qwen 得到“病变形状”，
    只对当前图片文件夹中“能找到图片”的那些行进行处理，
    写入 CSV 最后一列（列名 lesion_shape），并保存新 CSV。
    """
    config = load_config(config_path)
    api_config = config["api_config"]
    analysis_config = config["analysis_config"]
    prompts_config = config.get("prompts", {})

    folder_path = analysis_config["folder_path"]
    csv_file_path = analysis_config["csv_file_path"]
    supported_formats = [s.lower() for s in analysis_config["supported_formats"]]

    # image_id 列名：优先读 config，没有的话自动尝试“图片ID”
    image_id_column = analysis_config.get("image_id_column", None)

    shape_prompt = prompts_config.get(
        "shape_prompt",
        "你是一名专业的皮肤科医生。现在只根据皮肤病变图像本身，判断病变的大致几何形状。"
        "请只输出一个不超过10个字的中文短语，不要解释。"
    )

    # 1. 读取 CSV（自动处理编码）
    df = safe_read_csv(csv_file_path)
    print(f"📊 成功加载 CSV，记录数：{len(df)}")
    print(f"当前列名：{df.columns.tolist()}")

    # 如果 config 里没给 image_id_column，或者给的列不存在，就尝试“图片ID”
    if image_id_column is None or image_id_column not in df.columns:
        if "图片ID" in df.columns:
            image_id_column = "图片ID"
            print("ℹ️ 未在 config 中找到可用的 image_id_column，自动使用列名：图片ID")
        else:
            raise ValueError(
                f"在 CSV 中未找到配置的 image_id_column，也不存在“图片ID”这一列，请检查：现有列为 {df.columns.tolist()}"
            )

    print(f"👉 将使用列 {image_id_column!r} 作为图片 ID 列")

    # 新增一列，用于保存形状描述
    if "lesion_shape" not in df.columns:
        df["lesion_shape"] = ""
        print("🆕 新增列 lesion_shape 用于保存病变形状")
    else:
        print("ℹ️ 已存在列 lesion_shape，将在其基础上补全/覆盖空值")

    # 2. 构建 image_id -> image_path 映射（只来自当前 folder_path）
    imageid_to_path = build_imageid_to_path(folder_path, set(supported_formats))

    # 3. 只保留 image_id 在图片文件夹中能找到的行（方案一）
    #    也就是这一步把 1300 行里“没有对应图片”的全部过滤掉
    valid_mask = df[image_id_column].astype(str).isin(imageid_to_path.keys())
    df_to_process = df[valid_mask].copy()
    total_rows = len(df_to_process)

    print(f"👉 这次只处理在当前图片文件夹中能找到图片的行：{total_rows} 条")
    if total_rows == 0:
        print("⚠️ 当前 CSV 中没有任何 image_id 能在图片文件夹中找到对应图片，直接退出。")
        return

    # 4. 初始化 Qwen 客户端（多 key 轮询）
    api_keys = api_config["api_keys"]
    base_url = api_config["base_url"]
    model_type = api_config["model_type"]

    clients = [
        OpenAI(api_key=key, base_url=base_url)
        for key in api_keys
    ]
    num_clients = len(clients)
    print(f"🔑 共加载 {num_clients} 个 API key，将轮询使用")

    # 5. 逐行处理“有图片”的 subset
    for idx, (row_idx, row) in enumerate(df_to_process.iterrows()):
        image_id = str(row[image_id_column])

        # 如果这一行已经有 lesion_shape（例如断点续跑），就跳过
        if isinstance(df.at[row_idx, "lesion_shape"], str) and df.at[row_idx, "lesion_shape"].strip():
            continue

        image_path = imageid_to_path.get(image_id)
        if image_path is None:
            # 理论上不会出现，因为前面已经用 valid_mask 过滤过
            print(f"⚠️ [{idx+1}/{total_rows}] image_id={image_id} 在文件夹中找不到对应图片，标记为 image_not_found")
            df.at[row_idx, "lesion_shape"] = "image_not_found"
            continue

        client = clients[idx % num_clients]

        try:
            print(f"🩺 [{idx+1}/{total_rows}] image_id={image_id} -> {image_path.name}")
            shape = call_qwen_shape(client, model_type, image_path, shape_prompt)
            print(f"   形状：{shape}")
            # 写回原始 df 中对应行
            df.at[row_idx, "lesion_shape"] = shape

        except Exception as e:
            print(f"❌ 调用 Qwen 失败 (row {row_idx}, image_id={image_id}): {e}")
            df.at[row_idx, "lesion_shape"] = f"error:{e}"

        # 轻微限速，防止 QPS 过高
        time.sleep(0.2)

        # 每 N 行保存一次临时文件，防止中途断掉
        if (idx + 1) % save_every_n == 0:
            tmp_out = output_csv_path or csv_file_path.replace(".csv", "_with_shape.csv")
            df.to_csv(tmp_out, index=False, encoding="utf-8-sig")
            print(f"💾 已保存中间结果到 {tmp_out}")

    # 6. 最终保存
    final_out = output_csv_path or csv_file_path.replace(".csv", "_with_shape.csv")
    df.to_csv(final_out, index=False, encoding="utf-8-sig")
    print(f"\n🎉 所有图片处理完成！")
    print(f"📄 结果已保存到：{final_out}")


if __name__ == "__main__":
    print("🚀 启动皮肤病变“形状”标注程序（基于 Qwen-VL，方案一：只处理有图片的行）...")
    annotate_image_shapes("config3.json")
    print("✅ 全部完成。")
