import os
import pandas as pd
from pathlib import Path


def find_images_in_excel(folder_path, excel_path, output_path, search_column=1):
    """
    读取文件夹中的图片名，在Excel表格的第二列查找，输出匹配的行到新Excel

    参数:
    folder_path: 图片文件夹路径
    excel_path: 原始Excel文件路径
    output_path: 输出Excel文件路径
    search_column: 查找的列索引（从0开始，第二列是1）
    """

    try:
        # 1. 读取文件夹中的所有图片文件名（不含扩展名）
        image_files = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

        print("正在读取图片文件...")
        for file in os.listdir(folder_path):
            file_path = Path(file)
            if file_path.suffix.lower() in supported_formats:
                # 获取文件名（不含扩展名）
                image_files.append(file_path.stem)

        print(f"找到 {len(image_files)} 个图片文件")
        print("图片文件名示例:", image_files[:5])

        # 2. 读取Excel文件
        print("正在读取Excel文件...")
        df = pd.read_csv(excel_path,sep=',')
        print(f"Excel文件包含 {len(df)} 行, {len(df.columns)} 列")
        print("列名:", df.columns.tolist())

        # 检查查找列是否存在
        if search_column >= len(df.columns):
            print(f"错误: 查找列索引 {search_column} 超出范围，Excel只有 {len(df.columns)} 列")
            return

        column_name = df.columns[search_column]
        print(f"将在列 '{column_name}' 中查找图片名")

        # 3. 在指定列中查找匹配的图片名
        print("正在查找匹配项...")
        matched_rows = []

        for index, row in df.iterrows():
            cell_value = str(row[search_column]) if pd.notna(row[search_column]) else ""

            # 在图片文件名列表中查找
            for image_name in image_files:
                if image_name in cell_value:
                    matched_rows.append(row)
                    break

        # 4. 创建包含匹配行的新DataFrame
        if matched_rows:
            result_df = pd.DataFrame(matched_rows)
            print(f"找到 {len(result_df)} 个匹配项")

            # 5. 保存到新的Excel文件
            result_df.to_csv(output_path, index=False,sep=',')
            print(f"结果已保存到: {output_path}")

            # 显示部分结果
            print("\n前5个匹配项:")
            for i, row in result_df.head().iterrows():
                print(f"行 {i}: {row[search_column]}")

        else:
            print("未找到任何匹配项")
            # 创建空的DataFrame保存
            pd.DataFrame().to_excel(output_path, index=False)
            print(f"已创建空的结果文件: {output_path}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


# 更灵活的版本：支持精确匹配和模糊匹配
def find_images_in_excel_advanced(folder_path, excel_path, output_path,
                                  search_column=1, match_type='contains'):
    """
    高级版本：支持不同的匹配方式

    参数:
    folder_path: 图片文件夹路径
    excel_path: 原始Excel文件路径
    output_path: 输出Excel文件路径
    search_column: 查找的列索引
    match_type: 匹配方式 - 'contains', 'exact', 'startswith', 'endswith'
    """

    # 读取图片文件
    image_files = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    for file in os.listdir(folder_path):
        file_path = Path(file)
        if file_path.suffix.lower() in supported_formats:
            image_files.append(file_path.stem)

    print(f"找到 {len(image_files)} 个图片文件")

    # 读取Excel
    df = pd.read_csv(excel_path,sep=',')
    matched_rows = []

    for index, row in df.iterrows():
        cell_value = str(row[search_column]) if pd.notna(row[search_column]) else ""

        for image_name in image_files:
            found = False

            if match_type == 'exact' and cell_value == image_name:
                found = True
            elif match_type == 'contains' and image_name in cell_value:
                found = True
            elif match_type == 'startswith' and cell_value.startswith(image_name):
                found = True
            elif match_type == 'endswith' and cell_value.endswith(image_name):
                found = True

            if found:
                matched_rows.append(row)
                break

    # 保存结果
    if matched_rows:
        result_df = pd.DataFrame(matched_rows)
        result_df.to_csv(output_path, index=False,sep=',')
        print(f"找到 {len(result_df)} 个匹配项，已保存到: {output_path}")
    else:
        pd.DataFrame().to_excel(output_path, index=False)
        print("未找到匹配项，已创建空文件")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径 - 请根据实际情况修改这些路径
    folder_path = r"C:\Users\zhangrx59\Desktop\photos"  # 图片文件夹路径
    excel_path = r"C:\Users\zhangrx59\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2\HAM10000_metadata.csv"  # 原始Excel文件路径
    output_path = r"C:\Users\zhangrx59\Desktop\photos.csv"  # 输出文件路径

    # 使用方法1：基本版本
    print("=== 基本版本 ===")
    find_images_in_excel(folder_path, excel_path, output_path, search_column=1)

    print("\n=== 高级版本 ===")
    # 使用方法2：高级版本（精确匹配）
    output_path2 = r"C:\Users\zhangrx59\Desktop\photos.csv"
    find_images_in_excel_advanced(folder_path, excel_path, output_path2,
                                  search_column=1, match_type='exact')