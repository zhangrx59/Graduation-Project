import os
import shutil
import pandas as pd

# ======== 路径配置（按需修改） ========
CSV_PATH   = r"..\liandan\metadata_filtered.csv"  # 你的csv文件
IMAGE_DIR  = r"C:\Users\zhangrx59\.cache\kagglehub\datasets\mahdavi1202\skin-cancer\versions\1\imgs_part_1\imgs_part_1"  # 原始图片所在文件夹
OUTPUT_DIR = r"C:\Users\zhangrx59\.cache\kagglehub\datasets\mahdavi1202\skin-cancer\versions\1\imgs_part_1\selected_pics"  # 用来存放匹配到图片的空文件夹

# 支持的图片格式
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======== 读取第一列图片名，去掉扩展名 ========
df = pd.read_csv(CSV_PATH)
raw_names = df[df.columns[0]].astype(str).str.strip()

# 去掉扩展名（如 PAT_46_881_939.png -> PAT_46_881_939）
names_no_ext = raw_names.str.replace(r"\.\w+$", "", regex=True)

print(f"从CSV中读取到 {len(names_no_ext)} 个图片基名，示例：{names_no_ext[:5].tolist()}")

# ======== 建立图片目录的索引（不区分大小写） ========
image_files = {
    os.path.splitext(fname)[0].lower(): fname
    for fname in os.listdir(IMAGE_DIR)
    if os.path.splitext(fname)[1].lower() in IMAGE_EXTS
}

print(f"\n在图片目录中找到 {len(image_files)} 张图片")

# ======== 开始匹配并复制 ========
copied = 0
missing = []

for base_name in names_no_ext:
    key = base_name.lower()

    if key in image_files:
        src = os.path.join(IMAGE_DIR, image_files[key])
        dst = os.path.join(OUTPUT_DIR, image_files[key])
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing.append(base_name)

print(f"\n🎯 匹配结果：")
print(f"  成功复制：{copied} 张图片")
print(f"  未找到：{len(missing)} 张图片")

# 保存未匹配列表用于排查
if missing:
    missing_csv = os.path.join(os.path.dirname(CSV_PATH), "missing_images.csv")
    pd.Series(missing, name="missing_image_name").to_csv(missing_csv, index=False)
    print(f"\n未匹配的文件名已保存到: {missing_csv}")
    print("（可能原因：文件丢失/扩展名不一致/文件名有空格/文件已损坏）")

