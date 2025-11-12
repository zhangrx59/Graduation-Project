import os
import base64
import json
import pandas as pd
import multiprocessing
import time
import re
from pathlib import Path
from multiprocessing import Pool
from openai import OpenAI
from collections import Counter

 
def load_config(config_path="config.json"):
    """从JSON文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 {config_path} 未找到")
        raise
    except json.JSONDecodeError:
        print(f"错误：配置文件 {config_path} 格式不正确")
        raise


def encode_image_to_base64(image_path):
    """将本地图像转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_patient_data(csv_path):
    """
    加载患者临床数据，包括真实标签
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载患者数据，共 {len(df)} 条记录")
        print(f"数据列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"加载患者数据失败: {e}")
        return None


def create_multimodal_prompt(clinical_data, config):
    """
    从config中读取所有提示词组件，构建完整的多模态提示词
    """
    prompts_config = config["prompts"]

    clinical_info = ""
    if clinical_data is not None:
        clinical_info = "患者临床信息：\n"

        # 年龄
        if 'age' in clinical_data and pd.notna(clinical_data['age']):
            age = clinical_data['age']
            clinical_info += f"- 年龄: {age} 岁\n"

        # 性别
        if 'sex' in clinical_data and pd.notna(clinical_data['sex']):
            sex = clinical_data['sex']
            sex_display = "男性" if sex.lower() in ['male', 'm'] else "女性" if sex.lower() in ['female', 'f'] else sex
            clinical_info += f"- 性别: {sex_display}\n"

        # 病变部位
        if 'localization' in clinical_data and pd.notna(clinical_data['localization']):
            localization = clinical_data['localization']
            localization_mapping = {
                'back': '背部', 'lower extremity': '下肢', 'face': '面部',
                'trunk': '躯干', 'chest': '胸部', 'unknown': '未知部位',
                'upper extremity': '上肢', 'abdomen': '腹部', 'foot': '足部'
            }
            loc_display = localization_mapping.get(localization, localization)
            clinical_info += f"- 病变部位: {loc_display}\n"

    # 构建分析步骤
    analysis_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(prompts_config["analysis_steps"])])

    # 构建疾病类别
    disease_categories = "\n".join(
        [f"- {name} ({code})" for code, name in prompts_config["disease_categories"].items()])

    # 构建完整提示词 - 强调只输出病变类型
    full_prompt = f"""{prompts_config["base_prompt"]}

{clinical_info}

请基于以上临床信息和皮肤病变图像，进行综合分析：

分析步骤：
{analysis_steps}

可选诊断类别：
{disease_categories}

{prompts_config["output_requirement"]}

重要：请只输出最终的病变类型英文缩写，不要输出其他任何内容。"""

    return full_prompt


def find_patient_data_by_image_id(patient_df, image_filename, image_id_column='image_id'):
    """
    根据图片文件名查找对应的患者数据
    返回临床数据和真实标签，但真实标签不用于提示词
    """
    if patient_df is None:
        return None, None

    # 从图片文件名提取image_id（去掉扩展名）
    image_id = Path(image_filename).stem

    # 在image_id列中精确匹配
    if image_id_column in patient_df.columns:
        match = patient_df[patient_df[image_id_column] == image_id]
        if not match.empty:
            # 返回临床数据（用于提示词）和真实标签（用于比较）
            clinical_data = match.iloc[0][['age', 'sex', 'localization']].to_dict()
            true_label = match.iloc[0]['dx']  # 真实标签
            return clinical_data, true_label

    return None, None


def extract_diagnosis_from_response(response_text):
    """
    从大模型响应中提取诊断结果 - 保持使用dx:前缀
    """
    diagnosis_codes = ['dx:akiec', 'dx:bcc', 'dx:bkl', 'dx:df', 'dx:nv', 'dx:mel', 'dx:vasc']

    # 转换为小写便于匹配
    text_lower = response_text.lower()

    # 查找所有出现的诊断代码
    found_codes = []
    for code in diagnosis_codes:
        if re.search(r'\b' + code + r'\b', text_lower):
            found_codes.append(code)

    # 根据出现位置和频率判断主要诊断
    if len(found_codes) == 0:
        return "unknown"
    elif len(found_codes) == 1:
        return found_codes[0]
    else:
        # 如果有多个诊断，选择最后一个（通常是最终结论）
        return found_codes[-1]


def analyze_single_image_multimodal(args):
    """
    单张图片的多模态分析任务
    参数: (image_path, config, api_key, base_url, model_type, process_id, output_dir, patient_df, image_id_column)
    """
    image_path, config, api_key, base_url, model_type, process_id, output_dir, patient_df, image_id_column = args

    current_process = multiprocessing.current_process()
    pid = current_process.pid
    image_filename = image_path.name

    # 初始化输出文件
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"process_{process_id}_output.txt"

    print(f"⚡ [进程{process_id} PID:{pid}] 开始分析: {image_filename}")

    # 查找患者临床数据和真实标签
    clinical_data, true_label = find_patient_data_by_image_id(patient_df, image_filename, image_id_column)

    # 创建多模态提示词
    multimodal_prompt = create_multimodal_prompt(clinical_data, config)

    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 将图像转换为 base64
    base64_image = encode_image_to_base64(image_path)

    try:
        # 向输出文件写入开始信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"图片: {image_filename}\n")
            f.write(f"进程: {process_id} (PID: {pid})\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            if clinical_data:
                f.write(f"\n患者临床信息:\n")
                if 'age' in clinical_data and pd.notna(clinical_data['age']):
                    f.write(f"  年龄: {clinical_data['age']} 岁\n")
                if 'sex' in clinical_data and pd.notna(clinical_data['sex']):
                    sex_display = "男性" if clinical_data['sex'].lower() in ['male', 'm'] else "女性" if clinical_data[
                                                                                                             'sex'].lower() in [
                                                                                                             'female',
                                                                                                             'f'] else \
                        clinical_data['sex']
                    f.write(f"  性别: {sex_display}\n")
                if 'localization' in clinical_data and pd.notna(clinical_data['localization']):
                    loc_mapping = {
                        'back': '背部', 'lower extremity': '下肢', 'face': '面部',
                        'trunk': '躯干', 'chest': '胸部', 'unknown': '未知部位',
                        'upper extremity': '上肢', 'abdomen': '腹部', 'foot': '足部'
                    }
                    loc_display = loc_mapping.get(clinical_data['localization'], clinical_data['localization'])
                    f.write(f"  病变部位: {loc_display}\n")
            else:
                f.write(f"患者临床信息: 未找到对应数据\n")

            f.write(f"{'=' * 60}\n")

        # 向模型发送多模态请求
        response = client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": multimodal_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            stream=True,
        )

        # 收集响应内容
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content = delta.content
                full_response += content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning = delta.reasoning_content
                full_response += reasoning

        # 提取诊断结果（带dx:前缀）
        diagnosis_with_prefix = extract_diagnosis_from_response(full_response)

        # 去除dx:前缀，用于比较
        diagnosis = diagnosis_with_prefix.replace('dx:', '') if diagnosis_with_prefix.startswith(
            'dx:') else diagnosis_with_prefix

        # 判断诊断是否正确
        is_correct = (diagnosis == true_label) if true_label else False

        # 写入响应和诊断结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n大模型完整响应:\n{full_response}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write(f"提取的诊断结果(带前缀): {diagnosis_with_prefix}\n")
            f.write(f"用于比较的诊断代码: {diagnosis}\n")
            if true_label:
                f.write(f"真实标签: {true_label}\n")
                f.write(f"诊断是否正确: {'✅ 正确' if is_correct else '❌ 错误'}\n")
            else:
                f.write(f"真实标签: 未找到\n")
                f.write(f"诊断是否正确: 无法判断\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"\n分析完成\n")
            f.write(f"总响应长度: {len(full_response)} 字符\n")
            f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        print(
            f"✅ [进程{process_id} PID:{pid}] 完成分析: {image_filename} -> {diagnosis} (真实: {true_label}) {'✅' if is_correct else '❌'}")

        return {
            "image_name": image_filename,
            "process_id": process_id,
            "pid": pid,
            "clinical_data_found": clinical_data is not None,
            "clinical_data": clinical_data,
            "true_label": true_label,  # 新增真实标签
            "predicted_label": diagnosis,  # 新增预测标签（去除前缀）
            "predicted_label_with_prefix": diagnosis_with_prefix,  # 新增带前缀的预测标签
            "is_correct": is_correct,  # 新增正确性判断
            "response": full_response,
            "status": "success",
            "response_length": len(full_response),
            "output_file": str(output_file)
        }

    except Exception as e:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n分析失败\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"{'=' * 60}\n\n")

        print(f"❌ [进程{process_id} PID:{pid}] 分析失败: {image_filename} - {e}")
        return {
            "image_name": image_filename,
            "process_id": process_id,
            "pid": pid,
            "clinical_data_found": False,
            "clinical_data": None,
            "true_label": None,
            "predicted_label": "error",
            "predicted_label_with_prefix": "error",
            "is_correct": False,
            "response": "",
            "status": f"error: {str(e)}",
            "response_length": 0,
            "output_file": str(output_file)
        }


def analyze_skin_images_multimodal(config_path="config.json"):
    """
    多模态多进程批量分析
    """
    # 加载配置
    config = load_config(config_path)
    api_config = config["api_config"]
    analysis_config = config["analysis_config"]

    # 加载患者临床数据
    print("📊 加载患者临床数据...")
    patient_df = load_patient_data(analysis_config["csv_file_path"])

    if patient_df is None:
        print("❌ 无法加载患者数据，退出分析")
        return [], None

    # 获取文件夹中所有图片文件
    folder_path = analysis_config["folder_path"]
    supported_formats = analysis_config["supported_formats"]

    image_files = []
    for f in os.listdir(folder_path):
        file_path = Path(folder_path) / f
        if file_path.suffix.lower() in supported_formats and file_path.is_file():
            image_files.append(file_path)

    if not image_files:
        print("未在该文件夹中找到图片。")
        return [], patient_df

    print(f"共找到 {len(image_files)} 张图片")
    print(f"使用 {len(api_config['api_keys'])} 个API密钥进行多模态多进程分析...\n")

    # 准备任务参数
    tasks = []
    api_keys = api_config["api_keys"]
    output_dir = "multimodel_outputs_qwen"
    image_id_column = analysis_config.get("image_id_column", "image_id")

    for i, image_path in enumerate(image_files):
        # 轮询分配API密钥
        api_key = api_keys[i % len(api_keys)]
        process_id = i % len(api_keys) + 1

        task_args = (
            image_path,
            config,
            api_key,
            api_config["base_url"],
            api_config["model_type"],
            process_id,
            output_dir,
            patient_df,
            image_id_column
        )
        tasks.append(task_args)

    # 使用进程池执行任务
    max_workers = min(analysis_config.get("max_workers", 4), len(api_keys))
    results = []

    print(f"启动 {max_workers} 个进程进行分析...\n")
    start_time = time.time()

    with Pool(processes=max_workers) as pool:
        results = pool.map(analyze_single_image_multimodal, tasks)

    end_time = time.time()

    # 统计结果
    success_count = sum(1 for r in results if r["status"] == "success")
    clinical_data_found_count = sum(1 for r in results if r["clinical_data_found"])

    print(f"\n🎉 所有分析完成！")
    print(f"📊 总共处理: {len(results)} 张图片")
    print(f"✅ 成功: {success_count} 张")
    print(f"📋 找到临床数据: {clinical_data_found_count} 张")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")

    return results, patient_df


def create_multimodal_summary(results, patient_df, output_dir="multimodel_outputs_qwen"):
    """创建多模态分析汇总文件，包含诊断正确性分析"""
    summary_path = Path(output_dir) / "multimodal_analysis_summary.txt"

    # 计算准确率统计
    valid_results = [r for r in results if r["status"] == "success" and r["true_label"] is not None]
    correct_results = [r for r in valid_results if r["is_correct"]]

    accuracy = len(correct_results) / len(valid_results) if valid_results else 0

    # 各类别准确率统计
    category_stats = {}
    for result in valid_results:
        true_label = result["true_label"]
        predicted_label = result["predicted_label"]  # 使用去除前缀的版本进行比较
        is_correct = result["is_correct"]

        if true_label not in category_stats:
            category_stats[true_label] = {
                "total": 0,
                "correct": 0,
                "predictions": Counter()
            }

        category_stats[true_label]["total"] += 1
        category_stats[true_label]["predictions"][predicted_label] += 1

        if is_correct:
            category_stats[true_label]["correct"] += 1

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多模态皮肤图像分析汇总报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图片数量: {len(results)}\n")

        success_count = sum(1 for r in results if r["status"] == "success")
        clinical_count = sum(1 for r in results if r["clinical_data_found"])

        f.write(f"成功分析: {success_count} 张\n")
        f.write(f"结合临床数据: {clinical_count} 张\n")
        f.write(f"临床数据匹配率: {clinical_count / len(results) * 100:.1f}%\n\n")

        # 准确率统计
        f.write("诊断准确率统计:\n")
        f.write("-" * 40 + "\n")
        f.write(f"可评估图片数量: {len(valid_results)} 张\n")
        f.write(f"正确诊断数量: {len(correct_results)} 张\n")
        f.write(f"总体准确率: {accuracy:.2%}\n\n")

        # 各类别准确率
        f.write("各类别准确率详情:\n")
        f.write("=" * 80 + "\n")
        for true_label, stats in sorted(category_stats.items()):
            category_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"\n{true_label}:\n")
            f.write(f"  总数: {stats['total']} 张\n")
            f.write(f"  正确: {stats['correct']} 张\n")
            f.write(f"  准确率: {category_accuracy:.2%}\n")
            f.write(f"  预测分布: {dict(stats['predictions'])}\n")

        # 详细分析情况
        f.write("\n详细分析结果:\n")
        f.write("=" * 80 + "\n")

        for result in results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            clinical_icon = "📋" if result["clinical_data_found"] else "⚠️"

            f.write(f"\n{status_icon}{clinical_icon} {result['image_name']}\n")

            if result["status"] == "success":
                f.write(f"真实标签: {result['true_label'] if result['true_label'] else '未知'}\n")
                f.write(f"预测标签: {result['predicted_label']}\n")

                if result["true_label"]:
                    correctness_icon = "✅" if result["is_correct"] else "❌"
                    f.write(f"诊断结果: {correctness_icon} {'正确' if result['is_correct'] else '错误'}\n")
                else:
                    f.write(f"诊断结果: ⚠️ 无法判断正确性\n")

                if result["clinical_data_found"] and result["clinical_data"]:
                    clinical = result["clinical_data"]
                    f.write(f"临床信息: ")
                    if 'age' in clinical and pd.notna(clinical['age']):
                        f.write(f"年龄{clinical['age']}岁 ")
                    if 'sex' in clinical and pd.notna(clinical['sex']):
                        sex_display = "男" if clinical['sex'].lower() in ['male', 'm'] else "女"
                        f.write(f"{sex_display}性 ")
                    if 'localization' in clinical and pd.notna(clinical['localization']):
                        loc_mapping = {'back': '背部', 'lower extremity': '下肢', 'face': '面部', 'trunk': '躯干'}
                        loc_display = loc_mapping.get(clinical['localization'], clinical['localization'])
                        f.write(f"{loc_display}")
                    f.write("\n")

                f.write(f"处理进程: {result['process_id']}\n")
                f.write(f"响应长度: {result['response_length']} 字符\n")
            else:
                f.write(f"错误信息: {result['status']}\n")
            f.write("-" * 40 + "\n")

    print(f"📋 多模态分析汇总文件: {summary_path}")
    print(f"📊 诊断准确率: {accuracy:.2%} ({len(correct_results)}/{len(valid_results)})")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("🚀 启动多模态皮肤图像分析系统...")
    print("📊 将结合临床数据和图像进行综合分析...")
    print("🎯 大模型将只输出病变类型，系统将自动评估诊断准确率")

    results, patient_df = analyze_skin_images_multimodal("config.json")

    if results:
        create_multimodal_summary(results, patient_df)

    print("\n🎯 所有分析任务完成！")