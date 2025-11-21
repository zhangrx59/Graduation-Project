import pandas as pd

# ====== 路径设置 ======
input_csv = r"../liandan/metadata.csv"  # 你提供的原始文件路径
output_csv = r"../liandan/metadata_filtered.csv"

# 读取CSV文件
df = pd.read_csv(input_csv)

# ===== 条件①：第三列 & 第四列不全为空 =====
# df.columns[2] 表示第三列，df.columns[3] 表示第四列
df = df.dropna(subset=[df.columns[2], df.columns[3]], how='all')

# ===== 条件②：筛选第18列 =====
valid_labels = {"NEV", "BCC", "MEL", "SEK", "ACK"}
df = df[df[df.columns[17]].isin(valid_labels)]

# 保存处理后的文件
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"筛选完成！原始数据 {len(pd.read_csv(input_csv))} 行 → 过滤后 {len(df)} 行")
print(f"已保存到: {output_csv}")
