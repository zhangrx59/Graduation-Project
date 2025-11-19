import os
import shutil
import pandas as pd
from pathlib import Path


def copy_images_by_id_from_csv(
    csv_path,
    images_root,
    output_folder,
    id_column_index=1
):
    """
    从 CSV 的指定列读取图片 ID（按文件名，不含扩展名），
    在 images_root 下递归搜索所有图片文件，
    若图片名在 ID 列中，则复制到 output_folder。

    参数:
    csv_path: 包含图片 ID 的 CSV 文件路径
    images_root: 源图片根目录（会递归遍历）
    output_folder: 目标输出目录
    id_column_index: CSV 中 ID 所在列的索引（0 开始），第二列为 1
    """

    csv_path = Path(csv_path)
    images_root = Path(images_root)
    output_folder = Path(output_folder)

    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    # 1. 读取 CSV，获取第二列的图片 ID
    print(f"正在读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path, sep=',')
    print(f"CSV 共 {len(df)} 行, 列名: {df.columns.tolist()}")

    if id_column_index >= len(df.columns):
        raise IndexError(f"指定的列索引 {id_column_index} 超出范围，CSV 只有 {len(df.columns)} 列")

    # 取出指定列，并转为“去掉扩展名后的纯文件名”
    id_series = df.iloc[:, id_column_index].dropna().astype(str)

    # 比如单元格是 "ISIC_1234567.jpg" 或 "ISIC_1234567"
    # 都统一成 "ISIC_1234567"
    def normalize_id(x: str) -> str:
        return Path(x).stem

    id_set = {normalize_id(x) for x in id_series}
    print(f"从 CSV 中提取到 {len(id_set)} 个唯一图片 ID 示例: {list(id_set)[:5]}")

    # 2. 准备输出目录
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"目标输出目录: {output_folder}")

    # 3. 遍历源图片目录，找到匹配的文件并复制
    copied_count = 0
    visited_ids = set()

    print(f"正在遍历图片目录: {images_root}")
    for root, dirs, files in os.walk(images_root):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            suffix = file_path.suffix.lower()

            if suffix not in supported_formats:
                continue

            stem = file_path.stem  # 不含扩展名的文件名

            if stem in id_set:
                # 复制到目标目录
                dst_path = output_folder / file_path.name
                shutil.copy2(file_path, dst_path)
                copied_count += 1
                visited_ids.add(stem)

                print(f"复制: {file_path} -> {dst_path}")

    # 4. 总结
    print("\n===== 统计信息 =====")
    print(f"CSV 中的图片 ID 总数: {len(id_set)}")
    print(f"实际找到并复制的图片文件数: {copied_count}")

    missing_ids = id_set - visited_ids
    if missing_ids:
        print(f"有 {len(missing_ids)} 个 ID 在目录中未找到对应图片，示例: {list(missing_ids)[:10]}")
    else:
        print("所有 CSV 中的图片 ID 都找到了对应的图片文件。")


if __name__ == "__main__":
    # 你给的路径，直接写死
    csv_path = r"C:\Users\zhangrx59\Desktop\photos.csv"  # 包含图片 ID 的 CSV
    images_root = r"C:\Users\zhangrx59\PycharmProjects\liandan\kaggle\input\skin-cancer-mnist-ham10000"
    output_folder = r"C:\Users\zhangrx59\Desktop\photos"

    # 第二列 -> id_column_index = 1
    copy_images_by_id_from_csv(
        csv_path=csv_path,
        images_root=images_root,
        output_folder=output_folder,
        id_column_index=1
    )
