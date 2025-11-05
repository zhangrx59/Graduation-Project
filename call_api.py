import os
import base64
import json
import multiprocessing
import time
from pathlib import Path
from multiprocessing import Pool, Manager
from openai import OpenAI


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


def analyze_single_image(args):
    """
    单个图片分析任务 - 进程版本
    参数: (image_path, prompt, api_key, base_url, model_type, process_id, output_dir)
    """
    image_path, prompt, api_key, base_url, model_type, process_id, output_dir = args

    # 获取进程ID
    current_process = multiprocessing.current_process()
    pid = current_process.pid

    # 初始化输出文件
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"process_{process_id}_output.txt"

    # 在控制台显示进度
    print(f"⚡ [进程{process_id} PID:{pid}] 开始分析: {image_path.name}")

    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 将图像转换为 base64
    base64_image = encode_image_to_base64(image_path)

    try:
        # 向输出文件写入开始信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"🖼️ 图片: {image_path.name}\n")
            f.write(f"⚡ 进程: {process_id} (PID: {pid})\n")
            f.write(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n")

        # 向模型发送请求
        response = client.chat.completions.create(
            model=model_type,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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

        # 收集响应内容，只写入文件，不输出到控制台
        full_response = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content = delta.content
                full_response += content
                # 写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(content)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning = delta.reasoning_content
                full_response += reasoning
                # 写入文件
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(reasoning)

        # 写入结束信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n✅ 分析完成\n")
            f.write(f"📝 总响应长度: {len(full_response)} 字符\n")
            f.write(f"⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

        print(f"✅ [进程{process_id} PID:{pid}] 完成: {image_path.name}")

        return {
            "image_name": image_path.name,
            "process_id": process_id,
            "pid": pid,
            "response": full_response,
            "status": "success",
            "response_length": len(full_response),
            "output_file": str(output_file)
        }

    except Exception as e:
        # 写入错误信息
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n❌ 分析失败\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"⏰ 错误时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

        print(f"❌ [进程{process_id} PID:{pid}] 分析失败: {image_path.name} - {e}")
        return {
            "image_name": image_path.name,
            "process_id": process_id,
            "pid": pid,
            "response": "",
            "status": f"error: {str(e)}",
            "response_length": 0,
            "output_file": str(output_file)
        }


def analyze_skin_images_multiprocess(config_path="config.json"):
    """
    多进程批量分析文件夹中的皮肤图像
    """
    # 加载配置
    config = load_config(config_path)
    api_config = config["api_config"]
    analysis_config = config["analysis_config"]

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
        return []

    print(f"共找到 {len(image_files)} 张图片")
    print(f"使用 {len(api_config['api_keys'])} 个API密钥进行多进程分析...")
    print(f"大模型输出将保存到独立的txt文件中...\n")

    # 准备任务参数
    tasks = []
    api_keys = api_config["api_keys"]
    output_dir = "process_outputs"

    for i, image_path in enumerate(image_files):
        # 轮询分配API密钥
        api_key = api_keys[i % len(api_keys)]
        process_id = i % len(api_keys) + 1  # 进程ID从1开始

        task_args = (
            image_path,
            analysis_config["prompt"],
            api_key,
            api_config["base_url"],
            api_config["model_type"],
            process_id,
            output_dir
        )
        tasks.append(task_args)

    # 使用进程池执行任务
    max_workers = min(analysis_config.get("max_workers", 16), len(api_keys), 16)
    results = []

    print(f"启动 {max_workers} 个进程进行分析...\n")
    start_time = time.time()

    # 使用进程池
    with Pool(processes=max_workers) as pool:
        results = pool.map(analyze_single_image, tasks)

    end_time = time.time()

    # 显示输出文件信息
    print(f"\n📁 进程输出文件:")
    process_stats = {}
    for process_id in range(1, max_workers + 1):
        output_file = Path(output_dir) / f"process_{process_id}_output.txt"
        if output_file.exists():
            # 统计该进程处理的图片数量
            process_images = [r for r in results if r["process_id"] == process_id]
            success_count = sum(1 for r in process_images if r["status"] == "success")
            file_size = output_file.stat().st_size
            process_stats[process_id] = {
                "file": output_file,
                "total": len(process_images),
                "success": success_count,
                "size": file_size
            }
            print(f"  进程 {process_id}: {output_file}")
            print(f"     处理图片: {len(process_images)} 张, 成功: {success_count} 张")
            print(f"     文件大小: {file_size} 字节")

    # 统计总体结果
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    print(f"\n🎉 所有图片分析完成！")
    print(f"📊 总共处理: {len(results)} 张图片")
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {error_count} 张")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")
    print(f"🚀 平均速度: {len(results) / (end_time - start_time):.2f} 图片/秒")

    return results, process_stats


def create_summary_file(results, process_stats, output_dir="process_outputs"):
    """创建汇总文件"""
    summary_path = Path(output_dir) / "process_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多进程皮肤图像分析汇总\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总图片数量: {len(results)}\n")
        f.write(f"使用进程数量: {len(process_stats)}\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 按进程统计
        f.write("各进程处理情况:\n")
        f.write("-" * 40 + "\n")
        for process_id, stats in sorted(process_stats.items()):
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            f.write(f"进程 {process_id}:\n")
            f.write(f"  处理图片: {stats['success']}/{stats['total']} 成功\n")
            f.write(f"  成功率: {success_rate:.1f}%\n")
            f.write(f"  输出文件: {stats['file'].name}\n")
            f.write(f"  文件大小: {stats['size']} 字节\n\n")

        # 总体统计
        total_success = sum(stats["success"] for stats in process_stats.values())
        total_rate = (total_success / len(results)) * 100 if results else 0
        f.write(f"总体统计: {total_success}/{len(results)} 成功 ({total_rate:.1f}%)\n")

    print(f"📋 进程汇总文件: {summary_path}")


if __name__ == "__main__":
    # 在Windows上使用多进程需要这个保护
    multiprocessing.freeze_support()

    # === 使用多进程执行批量分析 ===
    print("🚀 启动16进程皮肤图像分析系统...")
    results, process_stats = analyze_skin_images_multiprocess("config.json")

    # 创建汇总文件
    if results:
        create_summary_file(results, process_stats)

    print("\n🎯 所有任务完成！")