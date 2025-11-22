# LoRA_medgamma_finetune.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

# ===================== 0. 配置区域 =====================

# 基础模型
BASE_MODEL = "google/medgemma-4b-it"

# 原始 metadata CSV（包含所有样本）
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 图片所在根目录 + 后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"   # 如果是 .jpg 就改成 ".jpg"

# 输出的划分后 CSV
TRAIN_CSV = METADATA_CSV.replace(".csv", "_train_5cls.csv")
VAL_CSV   = METADATA_CSV.replace(".csv", "_val_5cls.csv")
TEST_CSV  = METADATA_CSV.replace(".csv", "_test_5cls.csv")   # 用于后续 baseline / LoRA 评估

# 列名（与你的数据一致）
COL_IMAGE_ID    = "image_id"
COL_AGE         = "年龄"
COL_SEX         = "性别"
COL_FATHER_ORI  = "父籍贯"
COL_MOTHER_ORI  = "母籍贯"
COL_BIOPSY      = "是否活检"
COL_SMOKE       = "是否吸烟"
COL_DRINK       = "是否饮酒"
COL_PESTICIDE   = "农药"
COL_SKIN_CANCER = "皮肤癌病史"
COL_OTHER_CA    = "癌症病史"
COL_TAP_WATER   = "生活环境是否有自来水"
COL_SEWER       = "生活环境是否有下水道"
COL_PHOTOTYPE   = "皮肤光型"
COL_REGION      = "区域"
COL_D1          = "直径1"
COL_D2          = "直径2"
COL_PRURITUS    = "瘙痒"
COL_GROWTH      = "是否长大"
COL_PAIN        = "疼痛"
COL_MORPH_CHANGE= "形态变化"
COL_BLEEDING    = "出血"
COL_ELEVATED    = "是否隆起"

# 皮肤病分类标签列（dx / 诊断标签 等）
COL_TARGET      = "dx"

# 只保留这 5 类
ALLOWED_DX = ["akiec", "bcc", "bkl", "nev", "mel"]

# LoRA adapter 输出目录（调好的大模型就保存在这里）
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\medgemma_lora_derm_from_metadata"


# ===================== 1. 工具函数：病历摘要拼接 =====================

def yn_str(v, yes="有", no="无", unk="不详"):
    if isinstance(v, str):
        vs = v.strip().upper()
        if vs in ["TRUE", "T", "YES", "Y"]:
            return yes
        if vs in ["FALSE", "F", "NO", "N"]:
            return no
        if vs in ["UNK", "UNKNOWN", "NA", "NAN", "NONE", ""]:
            return unk
    if isinstance(v, (bool, int)):
        return yes if bool(v) else no
    if v != v:  # NaN
        return unk
    return str(v)


def build_clinical_note(row) -> str:
    """
    根据多列字段自动拼接成一段中文病历摘要文本。
    """
    age = row.get(COL_AGE, "")
    sex = str(row.get(COL_SEX, "") or "").strip()
    region = str(row.get(COL_REGION, "") or "").strip()
    father_ori = str(row.get(COL_FATHER_ORI, "") or "").strip()
    mother_ori = str(row.get(COL_MOTHER_ORI, "") or "").strip()

    skin_ca = yn_str(row.get(COL_SKIN_CANCER))
    other_ca = yn_str(row.get(COL_OTHER_CA))
    smoke = yn_str(row.get(COL_SMOKE), yes="吸烟", no="不吸烟")
    drink = yn_str(row.get(COL_DRINK), yes="饮酒", no="不饮酒")
    pesticide = yn_str(row.get(COL_PESTICIDE), yes="有农药接触史", no="无农药接触史")

    tap = yn_str(row.get(COL_TAP_WATER), yes="有自来水", no="无自来水")
    sewer = yn_str(row.get(COL_SEWER), yes="有下水道", no="无下水道")

    phototype = row.get(COL_PHOTOTYPE, "")
    d1 = row.get(COL_D1, "")
    d2 = row.get(COL_D2, "")

    pruritus = yn_str(row.get(COL_PRURITUS))
    growth = yn_str(row.get(COL_GROWTH))
    pain = yn_str(row.get(COL_PAIN))
    morph_change = yn_str(row.get(COL_MORPH_CHANGE))
    bleeding = yn_str(row.get(COL_BLEEDING))
    elevated = yn_str(row.get(COL_ELEVATED))

    # 性别汉化
    if isinstance(sex, str) and sex.upper() in ["MALE", "M"]:
        sex_cn = "男性"
    elif isinstance(sex, str) and sex.upper() in ["FEMALE", "F"]:
        sex_cn = "女性"
    else:
        sex_cn = sex or "性别不详"

    region_cn = region or "部位不详"

    size_str = ""
    if d1 and d2:
        size_str = f"皮损约 {d1}×{d2} mm"
    elif d1:
        size_str = f"皮损最大径约 {d1} mm"

    photo_str = f"皮肤光型：{phototype} 型" if phototype != "" else ""

    origin_str = ""
    if father_ori or mother_ori:
        origin_str = f"父籍贯：{father_ori}，母籍贯：{mother_ori}。"

    parts = []
    parts.append(f"{age}岁{sex_cn}，{region_cn}皮肤病变。")
    if size_str:
        parts.append(size_str + "。")
    if origin_str:
        parts.append(origin_str)

    parts.append(f"既往皮肤癌病史：{skin_ca}；其他恶性肿瘤病史：{other_ca}。")
    parts.append(f"生活方式：{smoke}，{drink}，{pesticide}。")
    parts.append(f"居住环境：{tap}，{sewer}。")
    if photo_str:
        parts.append(photo_str + "。")

    parts.append(
        f"症状体征：瘙痒{pruritus}，是否长大{growth}，疼痛{pain}，"
        f"形态变化{morph_change}，出血{bleeding}，隆起{elevated}。"
    )

    return "".join(parts)


def normalize_dx(label: str) -> str:
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()
    if s == "nv":
        s = "nev"
    return s


# ===================== 2. 按类别均匀划分 train/val/test（避免泄露） =====================

def prepare_splits(
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[str, str, str]:
    """
    从 METADATA_CSV 中：
    - 仅保留 dx ∈ ALLOWED_DX 的样本
    - 按类别 stratify 划分 train/val/test
    - 保存到 *_train_5cls.csv / *_val_5cls.csv / *_test_5cls.csv
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV):
        print("📁 发现已有划分文件，直接复用：")
        print(f"  train: {TRAIN_CSV}")
        print(f"  val  : {VAL_CSV}")
        print(f"  test : {TEST_CSV}")
        return TRAIN_CSV, VAL_CSV, TEST_CSV

    print(f"📄 读取原始 CSV: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV, encoding="utf-8")

    if COL_TARGET not in df.columns:
        raise ValueError(f"CSV 中找不到标签列 {COL_TARGET!r}")

    # 归一化 dx
    df["dx"] = df[COL_TARGET].apply(normalize_dx)
    df = df[df["dx"].isin(ALLOWED_DX)].copy()

    print(f"✅ 过滤后只保留 {ALLOWED_DX}，剩余样本数: {len(df)}")

    # 按 dx 分层划分 train / (val+test)
    df_train, df_tmp = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["dx"],
        random_state=seed,
    )

    # 再把 tmp 分成 val/test
    tmp_ratio = test_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_tmp,
        test_size=tmp_ratio,
        stratify=df_tmp["dx"],
        random_state=seed,
    )

    print("📊 按类别分层划分完成：")
    print("  train:", df_train["dx"].value_counts().to_dict())
    print("  val  :", df_val["dx"].value_counts().to_dict())
    print("  test :", df_test["dx"].value_counts().to_dict())

    # 保存
    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print("💾 已保存划分文件：")
    print(f"  train → {TRAIN_CSV}")
    print(f"  val   → {VAL_CSV}")
    print(f"  test  → {TEST_CSV}")

    return TRAIN_CSV, VAL_CSV, TEST_CSV


# ===================== 3. Dataset：病例 + 图像 → 分类标签 =====================

class DermMetadataDataset(Dataset):
    """
    基于划分后的 CSV 的 Dataset：
    - image: 由 图片ID + IMAGE_ROOT_DIR + IMAGE_EXT 拼路径
    - clinical_note: 由多列字段自动拼接
    - target_text: 皮肤病分类标签（akiec/bcc/bkl/nev/mel），作为生成目标
    """
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        image_id = str(row[COL_IMAGE_ID])
        image_path = os.path.join(IMAGE_ROOT_DIR, image_id + IMAGE_EXT)
        image = Image.open(image_path).convert("RGB")

        clinical_note = build_clinical_note(row)
        # 标签统一成小写字符串
        target_text = normalize_dx(str(row[COL_TARGET]))

        # 构造多模态对话：
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a dermatology assistant. "
                            "Given the clinical note and the skin lesion image, "
                            "your task is to classify the skin lesion into one of the following dx codes: "
                            "akiec, bcc, bkl, nev, mel. "
                            "Only output ONE code (akiec/bcc/bkl/nev/mel) as the final answer. "
                            "Do NOT output any other words or explanations."
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
                            "临床病历摘要如下：\n"
                            f"{clinical_note}\n\n"
                            "请结合病历和下方的皮肤病变图像，判断该病变最可能属于哪一类，"
                            "并只输出一个 dx 代码（akiec/bcc/bkl/nev/mel）作为答案："
                        ),
                    },
                    {"type": "image", "image": image},
                ],
            },
        ]

        return {
            "messages": messages,
            "image": image,
            "target_text": target_text,
        }


# ===================== 4. collator：AutoProcessor 打包多模态 =====================

@dataclass
class MedGemmaCollator:
    processor: AutoProcessor

    def __call__(self, batch) -> Dict[str, Any]:
        images = [eg["image"] for eg in batch]
        messages_list = [eg["messages"] for eg in batch]
        targets = [eg["target_text"] for eg in batch]

        texts = []
        for msgs, tgt in zip(messages_list, targets):
            chat_text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            # 把标签代码拼在后面，作为正确回答
            full_text = chat_text + tgt
            texts.append(full_text)

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


# ===================== 5. 加载模型 + LoRA（bf16，全精） =====================

def load_model_and_processor():
    print("🔧 加载 MedGemma 基础模型（bf16 + LoRA，全精权重，不用 bitsandbytes）...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,   # 如果 GPU 不支持 bf16，就改成 torch.float16 并在 TrainingArguments 里 fp16=True
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    # 开启梯度检查点（节省显存）
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    # 让输入需要 grad，方便 LoRA 训练
    model.enable_input_require_grads()

    # 挂载 LoRA 适配器
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"📊 总参数: {total/1e6:.1f}M, 可训练(LoRA): {trainable/1e6:.1f}M")

    return model, processor


# ===================== 6. 主训练入口 =====================

def main():
    # 1) 先按类别分层划分 train/val/test
    train_csv, val_csv, test_csv = prepare_splits()

    # 2) 用 HF Dataset 只加载 train/val（避免数据泄露）
    raw = load_dataset(
        "csv",
        data_files={"train": train_csv, "val": val_csv},
    )
    train_hf = raw["train"]
    val_hf = raw["val"]

    train_ds = DermMetadataDataset(train_hf)
    val_ds = DermMetadataDataset(val_hf)

    model, processor = load_model_and_processor()
    collator = MedGemmaCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,  # 如果不支持 bf16 就改成 fp16=True
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="steps",  # ← 旧版本 transformers 的写法
        eval_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA adapter 已保存到: {OUTPUT_DIR}")
    print(f"✅ 评估用的 test CSV 在: {TEST_CSV}")


if __name__ == "__main__":
    main()
