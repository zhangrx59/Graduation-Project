# finetune_medgemma_lora_from_metadata.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from datasets import load_dataset
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


# ===================== 0. 配置区域：你需要改的地方 =====================

# 基础模型
BASE_MODEL = "google/medgemma-4b-it"

# 你的 metadata CSV（就是 metadata_isic_with_shape.csv）
METADATA_CSV = r"C:\Users\zhangrx59\PycharmProjects\LoRA\metadata_isic_with_shape.csv"

# 图片所在根目录 + 后缀
IMAGE_ROOT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\ISIC_dataset"
IMAGE_EXT = ".png"   # 根据实际改成 ".jpg" 或 ".png"

# CSV 中列名（来自你表）
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

# 目标输出列：皮肤病分类标签（如 akiec/bcc/bkl/df/nv/mel/vasc）
# 如果你 CSV 里是中文列名，比如“诊断标签”，就改成 "诊断标签"
COL_TARGET      = "dx"

# LoRA adapter 输出目录（调好的模型就保存在这里）
OUTPUT_DIR = r"C:\Users\zhangrx59\PycharmProjects\LoRA\medgemma_lora_derm_from_metadata"


# ===================== 1. 把多列病历信息 -> 一段中文病历摘要 =====================

def yn_str(v, yes="有", no="无", unk="不详"):
    """
    把 True/False/'TRUE'/'FALSE'/'UNK'/NaN 统一转成中文 “有/无/不详”
    """
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
    根据表中的多列字段，自动拼接成一段中文病历摘要文本。
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

    # 部位
    region_cn = region or "部位不详"

    # 直径
    size_str = ""
    if d1 and d2:
        size_str = f"皮损约 {d1}×{d2} mm"
    elif d1:
        size_str = f"皮损最大径约 {d1} mm"

    # 皮肤光型
    photo_str = f"皮肤光型：{phototype} 型" if phototype != "" else ""

    # 出身地
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
        f"症状体征：瘙痒{pruritus}，是否长大{growth}，疼痛{pain}，形态变化{morph_change}，出血{bleeding}，隆起{elevated}。"
    )

    note = "".join(parts)
    return note


# ===================== 2. 数据集定义：根据病例 + 图像学习分类 =====================

class DermMetadataDataset(Dataset):
    """
    基于 metadata_isic_with_shape.csv 的 Dataset：
    - image: 由 图片ID + IMAGE_ROOT_DIR + IMAGE_EXT 拼路径
    - clinical_note: 由多列字段自动拼接
    - target_text: 皮肤病分类标签（如 akiec/bcc/bkl/df/nv/mel/vasc），作为生成目标
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
        target_text = str(row[COL_TARGET]).strip().lower()

        # 构造多模态对话：
        # system：告诉模型这是分类任务，只输出一个代码
        # user：给出中文病历摘要 + 图片
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
                            "akiec, bcc, bkl, df, nv, mel, vasc. "
                            "Only output ONE code (akiec/bcc/bkl/df/nv/mel/vasc) as the final answer. "
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
                            "并只输出一个 dx 代码（akiec/bcc/bkl/df/nv/mel/vasc）作为答案："
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


# ===================== 3. collator：用 AutoProcessor 统一打包多模态 =====================

@dataclass
class MedGemmaCollator:
    processor: AutoProcessor

    def __call__(self, batch) -> Dict[str, Any]:
        # 这里依赖 batch 里的 'image' / 'messages' / 'target_text'
        images = [eg["image"] for eg in batch]
        messages_list = [eg["messages"] for eg in batch]
        targets = [eg["target_text"] for eg in batch]

        texts = []
        for msgs, tgt in zip(messages_list, targets):
            # 把多轮对话 messages 展开成一段 chat 文本
            chat_text = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=False,
                tokenize=False,
            )
            # 直接把标签代码拼在后面，作为“正确回答”
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


# ===================== 4. 加载模型 + LoRA（不使用 bitsandbytes，bf16/FP16 全精） =====================

def load_model_and_processor():
    print("🔧 加载 MedGemma 基础模型（bf16 + LoRA，全精权重，不用bitsandbytes）...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,   # 如果报 bf16 不支持，就改成 torch.float16
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"

    # 开启梯度检查点（节省显存，可选）
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

    # 让模型输入需要 grad，这样 LoRA 层能正常训练
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


# ===================== 5. 主训练入口 =====================

def main():
    # 1) 读取 csv 到 HF Dataset
    raw = load_dataset("csv", data_files={"train": METADATA_CSV})["train"]

    # 然后再自己做 train/val 划分
    split = raw.train_test_split(test_size=0.1, seed=42)
    train_hf = split["train"]
    val_hf = split["test"]

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
        bf16=True,        # 如果 GPU 不支持 bf16，就改成：bf16=False, fp16=True
        fp16=False,
        report_to="none",  # 版本不支持的话可以改成 []
        remove_unused_columns=False,  # ☆ 关键：不要自动删掉 'image' / 'messages'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # 先放在这，将来可以手动 trainer.evaluate()
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ LoRA adapter 已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
