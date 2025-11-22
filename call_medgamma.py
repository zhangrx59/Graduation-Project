# -*- coding: utf-8 -*-
"""
使用 google/medgemma-4b-it 做多模态推理：
- 输入：一张皮肤病变图片 + 该图片对应的多列表格式“病历单”
- 病历单来自 CSV，多列共同构成临床信息
- label 列只作为真实标签 y*，不进入 prompt
- 输出：模型预测的 dx 类别代码（akiec/bcc/bkl/df/nv/mel/vasc）
- 计算总体准确率，并打印每个样本的预测结果

本版本做了批量推理优化：
- 按 BATCH_SIZE 组装多个样本，一次性送入 pipeline
- 让 batch_size 参数真正生效，提高 GPU 利用率
"""

import os
import re
import pandas as pd
from PIL import Image
from transformers import pipeline
import torch


# =============== 1. 路径 & 列名配置（你只需要改这里） =================

# 图片所在文件夹
IMAGE_FOLDER = r"C:/Users/zhangrx59/PycharmProjects/LoRA/ISIC_dataset"

# 病历 + 标签 CSV
CSV_PATH = r"C:/Users/zhangrx59/PycharmProjects/LoRA/metadata_isic_with_shape.csv"

# CSV 中标识图片的列：
# - 如果是 HAM10000 原始 metadata：通常是 "image_id"
# - 如果是你中文那种：“图片ID”
IMAGE_ID_COL = "image_id"      # 如果你的列叫“图片ID”，改成 "图片ID"

# CSV 中的真实标签列（只做 y*，不进 prompt）：
# - HAM10000 一般是 "dx"
# - 你给的另一张表是 "诊断标签"
LABEL_COL = "dx"               # 如果是中文表，改成 "诊断标签"

# 不作为病历文本输入的列（会自动排除）
EXCLUDE_COLS = {
    IMAGE_ID_COL,
    LABEL_COL,
    "lesion_shape",        # 如果有这列，通常是你生成的性状描述，可选
    "predicted_label",     # 如果你后面把预测结果写回 CSV，可以也排除
}

# 支持的图片后缀
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 批量大小（根据显存自行调整：4 / 8 / 16 ...）
BATCH_SIZE = 32


# =============== 2. 一些工具函数 =================

def safe_read_csv(path: str) -> pd.DataFrame:
    """尝试多种编码读取 CSV，防止中文编码问题。"""
    encodings = ["utf-8", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            print(f"尝试使用编码 {enc} 读取 {path} ...")
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"✅ 使用编码 {enc} 读取成功")
            return df
        except UnicodeDecodeError as e:
            print(f"❌ 编码 {enc} 失败: {e}")
            last_err = e
    raise last_err


def build_imageid_to_path(folder: str):
    """扫描图片文件夹，构建 {image_id(不含后缀) -> 完整路径} 映射。"""
    mapping = {}
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            image_id = os.path.splitext(name)[0]
            mapping[image_id] = p
    print(f"📁 在 {folder} 中共找到 {len(mapping)} 个图片文件（按 image_id 建索引）")
    return mapping


def build_clinical_text(row: pd.Series, clinical_cols):
    """
    把一行病历多列拼成一段文本。
    这里用「列名: 值」的方式，你觉得太啰嗦可以自己改成更顺的中文句子。
    """
    parts = []
    for col in clinical_cols:
        val = row.get(col, "")
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if not sval:
            continue
        parts.append(f"{col}: {sval}")
    return "；".join(parts)


def extract_dx_code(text: str):
    """
    从模型输出里提取 dx 代码，只允许以下几类（不区分大小写）：
    akiec, bcc, bkl, df, nv, mel, vasc
    """
    if not isinstance(text, str):
        return "unknown"
    text_lower = text.lower()
    m = re.search(r"\b(akiec|bcc|bkl|df|nv|mel|vasc)\b", text_lower)
    if m:
        return m.group(1)
    return "unknown"


# =============== 3. 加载 MedGEMMA 模型 =================

def load_medgemma_pipeline():
    print("🔧 正在加载 MedGEMMA 模型 google/medgemma-4b-it ...")
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("✅ 模型加载完成")
    return pipe


# =============== 4. 主评估逻辑（带批量推理优化） =================

def evaluate_medgemma_on_multimodal():
    # 1) 读 CSV
    df = safe_read_csv(CSV_PATH)

    # 检查必要列
    if IMAGE_ID_COL not in df.columns:
        raise ValueError(f"CSV 中找不到图片ID列 {IMAGE_ID_COL!r}，当前列名: {df.columns.tolist()}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV 中找不到标签列 {LABEL_COL!r}，当前列名: {df.columns.tolist()}")

    # 临床信息列 = 除去排除列以外的所有列
    clinical_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"📋 将以下列作为病历文本输入（不含 label 与图片ID）：\n{clinical_cols}")

    # 建立 image_id -> 文件路径 映射
    img_mapping = build_imageid_to_path(IMAGE_FOLDER)

    # 加载模型
    pipe = load_medgemma_pipeline()

    total = 0
    correct = 0
    missing_image = 0

    # 可选：把预测写回 DataFrame
    if "predicted_label" not in df.columns:
        df["predicted_label"] = ""

    # ---- 批量缓存 ----
    batch_messages = []   # 存放一批样本的 messages
    batch_meta = []       # 存 (idx, true_label, image_id)，方便和 outputs 对齐

    def flush_batch():
        """跑当前 batch，并把结果写回 df & 统计正确率。"""
        nonlocal total, correct
        if not batch_messages:
            return

        # 一次性跑一批
        outputs = pipe(
            text=batch_messages,
            max_new_tokens=32,
            batch_size=BATCH_SIZE,
        )

        # 对齐 meta 和输出
        for (idx, true_label, image_id), out in zip(batch_meta, outputs):
            # 这里沿用你原来的解析方式
            # out 的结构和单条调用一样，只是外面多了一层批次 list
            raw_text = out[0]["generated_text"][-1]["content"]
            pred_label = extract_dx_code(raw_text)

            df.at[idx, "predicted_label"] = pred_label

            is_correct = (pred_label == true_label)
            total += 1
            if is_correct:
                correct += 1

            print(
                f"🩺 [{total}] image_id={image_id} | pred={pred_label} | true={true_label} "
                f"| {'✅' if is_correct else '❌'} | raw: {raw_text!r}"
            )

        # 清空 batch，准备下一轮
        batch_messages.clear()
        batch_meta.clear()

    # ------------ 主循环：把样本按 batch 组装 ------------
    for idx, row in df.iterrows():
        image_id = str(row[IMAGE_ID_COL])
        true_label = str(row[LABEL_COL]).strip().lower()

        img_path = img_mapping.get(image_id)
        if img_path is None:
            print(f"⚠ image_id={image_id} 在文件夹中找不到图片，跳过")
            missing_image += 1
            continue

        clinical_text = build_clinical_text(row, clinical_cols)

        # 构造对话 prompt（注意：不包含 true_label）
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a dermatology AI assistant. "
                            "You will receive a patient's clinical information and a dermoscopy/skin image. "
                            "Your task is to predict the most likely lesion type using a single dx code."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the patient's clinical information (structured fields):\n"
                            f"{clinical_text}\n\n"
                            "Based ONLY on this information and the image, "
                            "predict the most likely skin lesion type.\n\n"
                            "Only output ONE of the following dx codes, and nothing else:\n"
                            "akiec, bcc, bkl, df, nv, mel, vasc"
                        ),
                    },
                    {
                        "type": "image",
                        "image": Image.open(img_path).convert("RGB"),
                    },
                ],
            },
        ]

        # 加入当前 batch
        batch_messages.append(messages)
        batch_meta.append((idx, true_label, image_id))

        # 如果凑够一个 batch，就推理一次
        if len(batch_messages) >= BATCH_SIZE:
            flush_batch()

    # 循环结束后，处理最后一个不足 BATCH_SIZE 的小 batch
    flush_batch()

    # ------------ 统计 & 保存结果 ------------
    print("\n====== 📊 评估结果 ======")
    print(f"有效样本数（有图片、参与评估）: {total}")
    print(f"缺少图片样本数: {missing_image}")
    if total > 0:
        acc = correct / total
        print(f"预测正确数: {correct}")
        print(f"总体准确率: {acc:.2%}")
    else:
        print("没有有效样本，无法计算准确率。")

    # 保存带 predicted_label 的 CSV（可选）
    out_path = CSV_PATH.replace(".csv", "_with_medgemma_pred.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"💾 已将预测结果写入: {out_path}")


if __name__ == "__main__":
    evaluate_medgemma_on_multimodal()
