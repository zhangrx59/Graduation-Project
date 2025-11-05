import os
import base64
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
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


class ThreadOutputManager:
    """线程输出管理器，每个线程有独立的输出文件"""

    def __init__(self, output_dir="thread_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.thread_files = {}
        self.lock = threading.Lock()

    def get_thread_file(self, thread_id):
        """获取线程对应的输出文件"""
        with self.lock:
            if thread_id not in self.thread_files:
                filename = self.output_dir / f"thread_{thread_id}_output.txt"
                self.thread_files[thread_id] = open(filename, 'w', encoding='utf-8')
                print(f"📄 创建线程 {thread_id} 的输出文件: {filename}")
            return self.thread_files[thread_id]

    def write_to_thread(self, thread_id, content):
        """向指定线程的输出文件写入内容"""
        file_obj = self.get_thread_file(thread_id)
        with self.lock:
            file_obj.write(content)
            file_obj.flush()

    def close_all(self):
        """关闭所有文件"""
        for file_obj in self.thread_files.values():
            file_obj.close()
        print("✅ 所有线程输出文件已关闭")


def analyze_single_image(args):
    """
    单个图片分析任务
    参数: (image_path, prompt, api_key, base_url, model_type, thread_id, output_manager)
    """
    image_path, prompt, api_key, base_url, model_type, thread_id, output_manager = args

    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 将图像转换为 base64
    base64_image = encode_image_to_base64(image_path)

    # 在控制台只显示进度，不显示大模型输出
    print(f"🧵 [线程{thread_id}] 开始分析: {image_path.name}")

    try:
        # 向输出文件写入开始信息
        start_msg = f"\n{'=' * 60}\n"
        start_msg += f"🖼️ 图片: {image_path.name}\n"
        start_msg += f"🧵 线程: {thread_id}\n"
        start_msg += f"{'=' * 60}\n"
        output_manager.write_to_thread(thread_id, start_msg)

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
                # 只写入文件，不输出到控制台
                output_manager.write_to_thread(thread_id, content)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning = delta.reasoning_content
                full_response += reasoning
                # 只写入文件，不输出到控制台
                output_manager.write_to_thread(thread_id, reasoning)

        # 写入结束信息
        end_msg = f"\n\n✅ 分析完成\n"
        end_msg += f"📝 总响应长度: {len(full_response)} 字符\n"
        end_msg += f"{'=' * 60}\n\n"
        output_manager.write_to_thread(thread_id, end_msg)

        print(f"✅ [线程{thread_id}] 完成: {image_path.name}")

        return {
            "image_name": image_path.name,
            "thread_id": thread_id,
            "response": full_response,
            "status": "success",
            "response_length": len(full_response)
        }

    except Exception as e:
        error_msg = f"\n❌ 分析失败\n"
        error_msg += f"错误信息: {str(e)}\n"
        error_msg += f"{'=' * 60}\n\n"
        output_manager.write_to_thread(thread_id, error_msg)

        print(f"❌ [线程{thread_id}] 分析失败: {image_path.name} - {e}")
        return {
            "image_name": image_path.name,
            "thread_id": thread_id,
            "response": "",
            "status": f"error: {str(e)}",
            "response_length": 0
        }


def analyze_skin_images_multithread(config_path="config.json"):
    """
    多线程批量分析文件夹中的皮肤图像
    """
    # 加载配置
    config = load_config(config_path)
    api_config = config["api_config"]
    analysis_config = config["analysis_config"]

    # 初始化输出管理器
    output_manager = ThreadOutputManager("thread_outputs")

    try:
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
        print(f"使用 {len(api_config['api_keys'])} 个API密钥进行多线程分析...")
        print(f"大模型输出将保存到独立的txt文件中（控制台不显示）...\n")

        # 准备任务参数
        tasks = []
        api_keys = api_config["api_keys"]

        for i, image_path in enumerate(image_files):
            # 轮询分配API密钥
            api_key = api_keys[i % len(api_keys)]
            thread_id = i % len(api_keys) + 1  # 线程ID从1开始

            task_args = (
                image_path,
                analysis_config["prompt"],
                api_key,
                api_config["base_url"],
                api_config["model_type"],
                thread_id,
                output_manager
            )
            tasks.append(task_args)

        # 使用线程池执行任务
        max_workers = min(analysis_config.get("max_workers", 4), len(api_keys))
        results = []

        print(f"启动 {max_workers} 个线程进行分析...\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(analyze_single_image, task): task for task in tasks}

            for future in future_to_task:
                result = future.result()
                results.append(result)

        # 显示输出文件信息
        print(f"\n📁 线程输出文件:")
        for thread_id in range(1, max_workers + 1):
            output_file = Path("thread_outputs") / f"thread_{thread_id}_output.txt"
            if output_file.exists():
                # 统计该线程处理的图片数量
                thread_images = [r for r in results if r["thread_id"] == thread_id]
                success_count = sum(1 for r in thread_images if r["status"] == "success")
                file_size = output_file.stat().st_size
                print(f"  线程 {thread_id}: {output_file}")
                print(f"     处理图片: {len(thread_images)} 张, 成功: {success_count} 张")
                print(f"     文件大小: {file_size} 字节")

        # 统计总体结果
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count

        print(f"\n🎉 所有图片分析完成！")
        print(f"📊 总共处理: {len(results)} 张图片")
        print(f"✅ 成功: {success_count} 张")
        print(f"❌ 失败: {error_count} 张")

        return results

    finally:
        # 确保关闭所有文件
        output_manager.close_all()


def create_summary_file(results, output_dir="thread_outputs"):
    """创建汇总文件，显示每个线程处理的图片"""
    summary_path = Path(output_dir) / "thread_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多线程皮肤图像分析汇总\n")
        f.write("=" * 50 + "\n\n")

        # 按线程分组
        thread_images = {}
        for result in results:
            thread_id = result["thread_id"]
            if thread_id not in thread_images:
                thread_images[thread_id] = []
            thread_images[thread_id].append({
                "image_name": result["image_name"],
                "status": result["status"],
                "response_length": result.get("response_length", 0)
            })

        for thread_id, images in sorted(thread_images.items()):
            f.write(f"线程 {thread_id} 处理的图片 ({len(images)} 张):\n")
            for img_info in images:
                status_icon = "✅" if img_info["status"] == "success" else "❌"
                f.write(f"  {status_icon} {img_info['image_name']}")
                if img_info["status"] == "success":
                    f.write(f" ({img_info['response_length']} 字符)")
                f.write(f" - {img_info['status']}\n")
            f.write("\n")

        # 总体统计
        total_success = sum(1 for r in results if r["status"] == "success")
        f.write(f"总体统计: {total_success}/{len(results)} 成功\n")

    print(f"📋 线程汇总文件: {summary_path}")


if __name__ == "__main__":
    # === 使用多线程执行批量分析 ===
    results = analyze_skin_images_multithread("config.json")

    # 创建汇总文件
    if results:
        create_summary_file(results)

    print("\n🎯 所有任务完成！")