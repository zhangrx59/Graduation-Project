import os, sys, json, hashlib, threading, traceback, base64, time, re
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from collections import Counter
from PIL import Image, ImageTk
import requests
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from openai import OpenAI

# ===================== 资源与用户数据路径 =====================
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

def app_data_dir():
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "ShowImageApp")
    os.makedirs(d, exist_ok=True)
    return d

USERS_FILE = os.path.join(app_data_dir(), "users.json")

# ===================== 登录/注册相关 =====================
def _hash(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        users = {"admin": _hash("123456")}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return users
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: str(v) for k, v in data.items()}
    except Exception:
        messagebox.showwarning("提示", "用户数据损坏，已重置。")
        users = {"admin": _hash("123456")}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return users

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def find_config_path(preferred: str | None = None) -> str:
    """
    返回可用的 config.json 绝对路径；找不到则抛异常。
    搜索顺序：
      1. preferred（手动指定）
      2. 当前工作目录
      3. EXE/脚本所在目录
      4. PyInstaller 解压目录（resource_path）
      5. 本地 AppData 文件夹
    """
    candidates = []
    if preferred:
        candidates.append(preferred)

    # 当前工作目录
    candidates.append(os.path.abspath("config.json"))

    # EXE/脚本所在目录
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(exe_dir, "config.json"))

    # PyInstaller 运行时解压目录
    try:
        base = getattr(sys, "_MEIPASS", None)
        if base:
            candidates.append(os.path.join(base, "config.json"))
    except Exception:
        pass

    # 本地 AppData 目录
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    candidates.append(os.path.join(base, "ShowImageApp", "config.json"))

    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError("未找到 config.json，请将它放在 EXE/脚本同目录或手动指定路径。")

# ===================== 统一配置读取与 Provider 访问 =====================
def load_config(config_path=None):
    """
    统一读取 config.json；若未指定路径，按 find_config_path 的搜索顺序查找。
    """
    if not config_path:
        config_path = find_config_path(None)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_provider_cfg(cfg: dict, provider: str) -> dict:
    """
    返回 providers.[provider] 的配置。兼容旧字段名 model_type -> model。
    """
    p = (cfg.get("providers") or {}).get(provider) or {}
    if "model" not in p and "model_type" in p:
        p["model"] = p["model_type"]
    return p

# ===================== Qwen/ChatGPT 简单对话（合并后版本） =====================
def _get_qwen_api_key_from_config_or_env() -> str:
    # 优先统一配置文件
    try:
        cfg = load_config(find_config_path(None))
        p = get_provider_cfg(cfg, "qwen")
        keys = p.get("api_keys", [])
        if isinstance(keys, list) and keys:
            return keys[0]
    except Exception:
        pass

    # 退回环境变量（按你原逻辑保留）
    for name in ("SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    raise RuntimeError("未找到 Qwen 的 API Key。请在 config.json 的 providers.qwen.api_keys[] 中提供，或设置环境变量（SILICONFLOW_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY）")

def _get_openai_api_key() -> str:
    # 1) 环境变量优先
    v = os.getenv("OPENAI_API_KEY")
    if v:
        return v

    # 2) 统一配置文件
    try:
        cfg = load_config(find_config_path(None))
        p = get_provider_cfg(cfg, "openai")
        keys = p.get("api_keys", [])
        if isinstance(keys, list) and keys:
            return keys[0]
    except Exception:
        pass

    raise RuntimeError("未找到 OPENAI_API_KEY。请在 config.json 的 providers.openai.api_keys[] 中提供，或设置环境变量 OPENAI_API_KEY。")

def call_model(provider: str, prompt: str) -> str:
    if provider == "qwen":
        api_key = _get_qwen_api_key_from_config_or_env()

        # 读取 provider 配置（允许被环境变量覆盖）
        try:
            cfg = load_config(find_config_path(None))
            pcfg = get_provider_cfg(cfg, "qwen")
        except Exception:
            pcfg = {}

        base = os.getenv("QWEN_BASE_URL", pcfg.get("base_url", "https://api.siliconflow.cn/v1"))
        model = os.getenv("QWEN_MODEL", pcfg.get("model", "Qwen/Qwen3-VL-235B-A22B-Thinking"))

        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        resp = requests.post(url, headers=headers, json=payload, timeout=pcfg.get("timeout", 60))
        if resp.status_code != 200:
            raise RuntimeError(f"Qwen 调用失败：HTTP {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    elif provider == "openai":
        api_key = _get_openai_api_key()

        # 读取 provider 配置（允许被环境变量覆盖）
        try:
            cfg = load_config(find_config_path(None))
            pcfg = get_provider_cfg(cfg, "openai")
        except Exception:
            pcfg = {}

        base = os.getenv("OPENAI_BASE_URL", pcfg.get("base_url", "https://www.dmxapi.com/v1"))
        model = os.getenv("OPENAI_MODEL", pcfg.get("model", "gpt-5-chat-latest"))

        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        resp = requests.post(url, headers=headers, json=payload, timeout=pcfg.get("timeout", 60))
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI 调用失败：HTTP {resp.status_code} {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    else:
        raise ValueError("未知 provider（应为 'qwen' 或 'openai'）")

# ===================== 多模态批量分析逻辑（基于统一配置） =====================
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_patient_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载患者数据，共 {len(df)} 条记录")
        print(f"数据列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"加载患者数据失败: {e}")
        return None

def create_multimodal_prompt(clinical_data, config):
    prompts_config = config["prompts"]

    clinical_info = ""
    if clinical_data is not None:
        clinical_info = "患者临床信息：\n"
        if 'age' in clinical_data and pd.notna(clinical_data['age']):
            clinical_info += f"- 年龄: {clinical_data['age']} 岁\n"
        if 'sex' in clinical_data and pd.notna(clinical_data['sex']):
            sex = clinical_data['sex']
            sex_display = "男性" if str(sex).lower() in ['male', 'm'] else "女性" if str(sex).lower() in ['female','f'] else sex
            clinical_info += f"- 性别: {sex_display}\n"
        if 'localization' in clinical_data and pd.notna(clinical_data['localization']):
            localization_mapping = {
                'back':'背部','lower extremity':'下肢','face':'面部','trunk':'躯干','chest':'胸部','unknown':'未知部位',
                'upper extremity':'上肢','abdomen':'腹部','foot':'足部'
            }
            loc_display = localization_mapping.get(clinical_data['localization'], clinical_data['localization'])
            clinical_info += f"- 病变部位: {loc_display}\n"

    analysis_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(prompts_config["analysis_steps"])])
    disease_categories = "\n".join([f"- {name} ({code})" for code,name in prompts_config["disease_categories"].items()])

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
    if patient_df is None:
        return None, None
    image_id = Path(image_filename).stem
    if image_id_column in patient_df.columns:
        match = patient_df[patient_df[image_id_column] == image_id]
        if not match.empty:
            clinical_data = match.iloc[0][['age','sex','localization']].to_dict()
            true_label  = match.iloc[0]['dx']
            return clinical_data, true_label
    return None, None

def extract_diagnosis_from_response(response_text):
    diagnosis_codes = ['dx:akiec','dx:bcc','dx:bkl','dx:df','dx:nv','dx:mel','dx:vasc']
    text_lower = (response_text or "").lower()
    found = []
    for code in diagnosis_codes:
        if re.search(r'\b' + code + r'\b', text_lower):
            found.append(code)
    if not found:
        return "unknown"
    return found[-1] if len(found) > 1 else found[0]

def analyze_single_image_multimodal(args):
    image_path, config, api_key, base_url, model_type, process_id, output_dir, patient_df, image_id_column = args
    current_process = multiprocessing.current_process()
    pid = current_process.pid
    image_filename = image_path.name

    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"process_{process_id}_output.txt"

    print(f"⚡ [进程{process_id} PID:{pid}] 开始分析: {image_filename}")

    clinical_data, true_label = find_patient_data_by_image_id(patient_df, image_filename, image_id_column)
    multimodal_prompt = create_multimodal_prompt(clinical_data, config)

    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_image_to_base64(image_path)

    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n图片: {image_filename}\n进程: {process_id} (PID: {pid})\n开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if clinical_data:
                f.write("\n患者临床信息:\n")
                if 'age' in clinical_data and pd.notna(clinical_data['age']): f.write(f"  年龄: {clinical_data['age']} 岁\n")
                if 'sex' in clinical_data and pd.notna(clinical_data['sex']):
                    sex_display = "男性" if str(clinical_data['sex']).lower() in ['male','m'] else "女性" if str(clinical_data['sex']).lower() in ['female','f'] else clinical_data['sex']
                    f.write(f"  性别: {sex_display}\n")
                if 'localization' in clinical_data and pd.notna(clinical_data['localization']):
                    loc_mapping={'back':'背部','lower extremity':'下肢','face':'面部','trunk':'躯干','chest':'胸部','unknown':'未知部位','upper extremity':'上肢','abdomen':'腹部','foot':'足部'}
                    loc_display = loc_mapping.get(clinical_data['localization'], clinical_data['localization'])
                    f.write(f"  病变部位: {loc_display}\n")
            else:
                f.write("患者临床信息: 未找到对应数据\n")
            f.write(f"{'='*60}\n")

        response = client.chat.completions.create(
            model=model_type,
            messages=[{
                "role":"user",
                "content":[
                    {"type":"text","text": multimodal_prompt},
                    {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            stream=True,
        )

        full_response = ""
        for chunk in response:
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None): full_response += delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content: full_response += delta.reasoning_content

        diagnosis_with_prefix = extract_diagnosis_from_response(full_response)
        diagnosis = diagnosis_with_prefix.replace('dx:','') if diagnosis_with_prefix.startswith('dx:') else diagnosis_with_prefix
        is_correct = (diagnosis == true_label) if true_label else False

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n大模型完整响应:\n{full_response}\n")
            f.write(f"\n{'='*60}\n提取的诊断结果(带前缀): {diagnosis_with_prefix}\n用于比较的诊断代码: {diagnosis}\n")
            if true_label:
                f.write(f"真实标签: {true_label}\n诊断是否正确: {'✅ 正确' if is_correct else '❌ 错误'}\n")
            else:
                f.write("真实标签: 未找到\n诊断是否正确: 无法判断\n")
            f.write(f"{'='*60}\n\n分析完成\n总响应长度: {len(full_response)} 字符\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        print(f"✅ [进程{process_id} PID:{pid}] 完成分析: {image_filename} -> {diagnosis} (真实: {true_label}) {'✅' if is_correct else '❌'}")

        return {
            "image_name": image_filename, "process_id": process_id, "pid": pid,
            "clinical_data_found": clinical_data is not None, "clinical_data": clinical_data,
            "true_label": true_label, "predicted_label": diagnosis,
            "predicted_label_with_prefix": diagnosis_with_prefix,
            "is_correct": is_correct, "response": full_response,
            "status": "success", "response_length": len(full_response),
            "output_file": str(output_file)
        }

    except Exception as e:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n分析失败\n错误信息: {str(e)}\n{'='*60}\n\n")
        print(f"❌ [进程{process_id} PID:{pid}] 分析失败: {image_filename} - {e}")
        return {
            "image_name": image_filename, "process_id": process_id, "pid": pid,
            "clinical_data_found": False, "clinical_data": None, "true_label": None,
            "predicted_label": "error", "predicted_label_with_prefix": "error",
            "is_correct": False, "response": "", "status": f"error: {str(e)}",
            "response_length": 0, "output_file": str(output_file)
        }

def analyze_skin_images_multimodal(config_path="config.json"):
    config = load_config(config_path)
    # 批量分析使用 Qwen 的 provider 配置
    api_config = get_provider_cfg(config, "qwen")
    analysis_config = config["analysis_config"]

    print("📊 加载患者临床数据...")
    patient_df = load_patient_data(analysis_config["csv_file_path"])
    if patient_df is None:
        print("❌ 无法加载患者数据，退出分析")
        return [], None

    folder_path = analysis_config["folder_path"]
    supported_formats = analysis_config["supported_formats"]

    image_files = []
    for f in os.listdir(folder_path):
        fp = Path(folder_path) / f
        if fp.suffix.lower() in supported_formats and fp.is_file():
            image_files.append(fp)
    if not image_files:
        print("未在该文件夹中找到图片。")
        return [], patient_df

    print(f"共找到 {len(image_files)} 张图片")
    print(f"使用 {len(api_config.get('api_keys', []))} 个API密钥进行多模态多进程分析...\n")

    tasks = []
    api_keys = api_config.get("api_keys", [])
    output_dir = "multimodel_outputs_qwen"
    image_id_column = analysis_config.get("image_id_column", "image_id")

    if not api_keys:
        raise RuntimeError("providers.qwen.api_keys 为空，请在 config.json 中提供至少一把 API Key。")

    for i, image_path in enumerate(image_files):
        api_key = api_keys[i % len(api_keys)]
        process_id = i % len(api_keys) + 1
        tasks.append((image_path, config, api_key, api_config["base_url"], api_config["model"],
                      process_id, output_dir, patient_df, image_id_column))

    max_workers = min(analysis_config.get("max_workers", 4), len(api_keys))
    print(f"启动 {max_workers} 个进程进行分析...\n")
    start = time.time()
    results = []
    with Pool(processes=max_workers) as pool:
        results = pool.map(analyze_single_image_multimodal, tasks)
    end = time.time()

    success_count = sum(1 for r in results if r["status"] == "success")
    clinical_data_found_count = sum(1 for r in results if r["clinical_data_found"])
    print(f"\n🎉 所有分析完成！")
    print(f"📊 总共处理: {len(results)} 张图片")
    print(f"✅ 成功: {success_count} 张")
    print(f"📋 找到临床数据: {clinical_data_found_count} 张")
    print(f"⏱️ 总耗时: {end - start:.2f} 秒")

    return results, patient_df

def create_multimodal_summary(results, patient_df, output_dir="multimodel_outputs_qwen"):
    summary_path = Path(output_dir) / "multimodal_analysis_summary.txt"
    valid_results = [r for r in results if r["status"] == "success" and r["true_label"] is not None]
    correct_results = [r for r in valid_results if r["is_correct"]]
    accuracy = len(correct_results) / len(valid_results) if valid_results else 0

    category_stats = {}
    for r in valid_results:
        true_label = r["true_label"]; predicted_label = r["predicted_label"]; is_correct = r["is_correct"]
        if true_label not in category_stats:
            category_stats[true_label] = {"total":0, "correct":0, "predictions": Counter()}
        category_stats[true_label]["total"] += 1
        category_stats[true_label]["predictions"][predicted_label] += 1
        if is_correct: category_stats[true_label]["correct"] += 1

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多模态皮肤图像分析汇总报告\n" + "="*80 + "\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图片数量: {len(results)}\n")
        success_count = sum(1 for r in results if r["status"] == "success")
        clinical_count = sum(1 for r in results if r["clinical_data_found"])
        f.write(f"成功分析: {success_count} 张\n")
        f.write(f"结合临床数据: {clinical_count} 张\n")
        f.write(f"临床数据匹配率: {clinical_count / len(results) * 100:.1f}%\n\n")
        f.write("诊断准确率统计:\n" + "-"*40 + "\n")
        f.write(f"可评估图片数量: {len(valid_results)} 张\n")
        f.write(f"正确诊断数量: {len(correct_results)} 张\n")
        f.write(f"总体准确率: {accuracy:.2%}\n\n")
        f.write("各类别准确率详情:\n" + "="*80 + "\n")
        for true_label, stats in sorted(category_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"]>0 else 0
            f.write(f"\n{true_label}:\n  总数: {stats['total']} 张\n  正确: {stats['correct']} 张\n  准确率: {acc:.2%}\n  预测分布: {dict(stats['predictions'])}\n")
        f.write("\n详细分析结果:\n" + "="*80 + "\n")
        for r in results:
            status_icon = "✅" if r["status"] == "success" else "❌"
            clinical_icon = "📋" if r["clinical_data_found"] else "⚠️"
            f.write(f"\n{status_icon}{clinical_icon} {r['image_name']}\n")
            if r["status"] == "success":
                f.write(f"真实标签: {r['true_label'] if r['true_label'] else '未知'}\n")
                f.write(f"预测标签: {r['predicted_label']}\n")
                if r["true_label"]:
                    correctness_icon = "✅" if r["is_correct"] else "❌"
                    f.write(f"诊断结果: {correctness_icon} {'正确' if r['is_correct'] else '错误'}\n")
                else:
                    f.write("诊断结果: ⚠️ 无法判断正确性\n")
                if r["clinical_data_found"] and r["clinical_data"]:
                    c = r["clinical_data"]; f.write("临床信息: ")
                    if 'age' in c and pd.notna(c['age']): f.write(f"年龄{c['age']}岁 ")
                    if 'sex' in c and pd.notna(c['sex']):
                        sex_display = "男" if str(c['sex']).lower() in ['male','m'] else "女"
                        f.write(f"{sex_display}性 ")
                    if 'localization' in c and pd.notna(c['localization']):
                        loc_map={'back':'背部','lower extremity':'下肢','face':'面部','trunk':'躯干'}
                        loc_display = loc_map.get(c['localization'], c['localization']); f.write(f"{loc_display}")
                    f.write("\n")
                f.write(f"处理进程: {r['process_id']}\n响应长度: {r['response_length']} 字符\n")
            else:
                f.write(f"错误信息: {r['status']}\n")
            f.write("-"*40 + "\n")

    print(f"📋 多模态分析汇总文件: {summary_path}")
    print(f"📊 诊断准确率: {accuracy:.2%} ({len(correct_results)}/{len(valid_results)})")
    return str(summary_path)

# ===================== GUI：App / Login / Register / Image 页 =====================
class App(tk.Tk):
    def __init__(self, image_rel_path):
        super().__init__()
        self.title("登录示例")
        self.geometry("1000x680")
        self.minsize(640, 400)

        self.image_rel_path = image_rel_path
        self.users = load_users()
        self.current_user = None

        self.login_frame = LoginFrame(self, on_success=self.show_image,
                                      on_register=self.open_register, users_provider=lambda: self.users)
        self.image_frame = ImageFrame(self, image_rel_path=self.image_rel_path,
                                      on_logout=self.show_login, on_exit=self.exit_app)

        self.login_frame.pack(fill="both", expand=True)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _e=None):
        if self.image_frame.winfo_ismapped():
            self.image_frame.render()

    def show_image(self, username: str):
        self.current_user = username
        self.login_frame.pack_forget()
        self.image_frame.pack(fill="both", expand=True)
        self.image_frame.render()
        self.title(f"显示图片 - {self.current_user}")

    def show_login(self):
        self.current_user = None
        self.image_frame.pack_forget()
        self.login_frame.pack(fill="both", expand=True)
        self.title("登录示例")

    def exit_app(self):
        self.destroy()

    def open_register(self):
        RegisterWindow(self, on_created=self._on_user_created, users_provider=lambda: self.users)

    def _on_user_created(self, username: str, password: str):
        u = username.strip()
        if not u:
            messagebox.showerror("注册失败", "用户名不能为空"); return
        if u in self.users:
            messagebox.showerror("注册失败", "该用户名已存在"); return
        if len(password) < 6:
            messagebox.showerror("注册失败", "密码长度至少 6 位"); return
        self.users[u] = _hash(password)
        save_users(self.users)
        messagebox.showinfo("成功", f"用户 '{u}' 注册成功，可使用新账户登录。")

class LoginFrame(tk.Frame):
    def __init__(self, master: App, on_success, on_register, users_provider):
        super().__init__(master, padx=16, pady=16)
        self.on_success = on_success
        self.on_register = on_register
        self.users_provider = users_provider

        logo_path = resource_path("assets/picture.jpg")
        self.tk_logo = None
        if os.path.exists(logo_path):
            try:
                _img = Image.open(logo_path); _img.thumbnail((260, 180))
                self.tk_logo = ImageTk.PhotoImage(_img)
                tk.Label(self, image=self.tk_logo).grid(row=0, column=0, columnspan=2, pady=(0, 10))
            except Exception:
                pass

        tk.Label(self, text="用户名").grid(row=1, column=0, sticky="e", pady=4, padx=(0, 6))
        self.ent_user = tk.Entry(self, width=24); self.ent_user.grid(row=1, column=1, pady=4); self.ent_user.insert(0, "admin")
        tk.Label(self, text="密码").grid(row=2, column=0, sticky="e", pady=4, padx=(0, 6))
        self.var_pwd = tk.StringVar()
        self.ent_pwd = tk.Entry(self, textvariable=self.var_pwd, width=24, show="•"); self.ent_pwd.grid(row=2, column=1, pady=4); self.ent_pwd.insert(0, "123456")

        self.var_show = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="显示密码", variable=self.var_show,
                       command=lambda: self.ent_pwd.config(show="" if self.var_show.get() else "•")
        ).grid(row=3, column=1, sticky="w", pady=(0, 6))

        btn_area = tk.Frame(self); btn_area.grid(row=4, column=0, columnspan=2, pady=8)
        tk.Button(btn_area, text="登录", width=12, command=self.try_login).pack(side="left", padx=6)
        tk.Button(btn_area, text="注册", width=12, command=self.on_register).pack(side="left", padx=6)

        self.bind_all("<Return>", lambda e: self.try_login())
        for c in range(2): self.grid_columnconfigure(c, weight=1)

    def try_login(self):
        user = self.ent_user.get().strip(); pwd = self.ent_pwd.get()
        users = self.users_provider()
        if user in users and users[user] == _hash(pwd):
            self.on_success(user)
        else:
            messagebox.showerror("登录失败", "用户名或密码错误")

class RegisterWindow(tk.Toplevel):
    def __init__(self, master: App, on_created, users_provider):
        super().__init__(master); self.title("注册"); self.resizable(False, False)
        self.on_created = on_created; self.users_provider = users_provider
        frm = tk.Frame(self, padx=16, pady=16); frm.grid(row=0, column=0)

        tk.Label(frm, text="用户名").grid(row=0, column=0, sticky="e", pady=4, padx=(0,6))
        self.ent_user = tk.Entry(frm, width=24); self.ent_user.grid(row=0, column=1, pady=4)
        tk.Label(frm, text="密码").grid(row=1, column=0, sticky="e", pady=4, padx=(0,6))
        self.ent_pwd = tk.Entry(frm, width=24, show="•"); self.ent_pwd.grid(row=1, column=1, pady=4)
        tk.Label(frm, text="确认密码").grid(row=2, column=0, sticky="e", pady=4, padx=(0,6))
        self.ent_pwd2 = tk.Entry(frm, width=24, show="•"); self.ent_pwd2.grid(row=2, column=1, pady=4)
        tk.Button(frm, text="创建账户", width=20, command=self.create_user).grid(row=3, column=0, columnspan=2, pady=10)
        self.bind("<Return>", lambda e: self.create_user())
        self.after(50, self._center)

    def _center(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        sw,sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x,y = (sw-w)//2, (sh-h)//3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def create_user(self):
        user = self.ent_user.get().strip()
        pwd1 = self.ent_pwd.get(); pwd2 = self.ent_pwd2.get()
        users = self.users_provider()
        if not user: messagebox.showerror("错误","用户名不能为空"); return
        if user in users: messagebox.showerror("错误","该用户名已存在"); return
        if len(pwd1) < 6: messagebox.showerror("错误","密码长度至少 6 位"); return
        if pwd1 != pwd2: messagebox.showerror("错误","两次输入的密码不一致"); return
        self.on_created(user, pwd1); self.destroy()

class ImageFrame(tk.Frame):
    def __init__(self, master, image_rel_path, on_logout, on_exit):
        super().__init__(master)
        self.on_logout = on_logout; self.on_exit = on_exit

        topbar = tk.Frame(self, bg="#444444", height=44); topbar.pack(fill="x", side="top")
        tk.Button(topbar, text="调用模型1（Qwen）", command=lambda: self.open_chat("qwen")).pack(side="left", padx=6, pady=5)
        tk.Button(topbar, text="调用模型2（ChatGPT）", command=lambda: self.open_chat("openai")).pack(side="left", padx=6, pady=5)
        tk.Button(topbar, text="批量分析（Qwen 皮肤图像）", command=self.open_batch).pack(side="left", padx=12, pady=5)

        tk.Button(topbar, text="退出登录", command=self.logout).pack(side="right", padx=10, pady=5)
        tk.Button(topbar, text="退出程序", command=self.exit_program).pack(side="right", padx=10, pady=5)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#222222"); self.canvas.pack(fill="both", expand=True)

        path = resource_path(image_rel_path)
        self.original = Image.open(path).convert("RGBA"); self.tk_img = None

    def render(self):
        if not hasattr(self, "original"): return
        cw = max(self.canvas.winfo_width(), 1); ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.original.size; scale = min(cw/iw, ch/ih)
        new_w = max(int(iw*scale),1); new_h = max(int(ih*scale),1)
        img_resized = self.original.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        x = (cw - new_w)//2; y = (ch - new_h)//2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

    def logout(self):
        if messagebox.askyesno("确认", "确定要退出登录吗？"): self.on_logout()

    def exit_program(self):
        if messagebox.askyesno("确认", "确定要退出程序吗？"): self.on_exit()

    def open_chat(self, provider: str):
        ChatDialog(self, provider=provider)

    def open_batch(self):
        BatchAnalyzeDialog(self)

# ===================== 对话框：Qwen / OpenAI =====================
class ChatDialog(tk.Toplevel):
    def __init__(self, master, provider: str):
        super().__init__(master); self.provider = provider
        self.title("Qwen 对话" if provider=="qwen" else "ChatGPT 对话")
        self.geometry("640x480"); self.resizable(True, True)

        top = tk.Frame(self, padx=8, pady=8); top.pack(fill="both", expand=True)
        tk.Label(top, text="输入提示词：").pack(anchor="w")
        self.txt_in = tk.Text(top, height=6); self.txt_in.pack(fill="x", expand=False)
        self.txt_in.insert("1.0", "帮我写一首小诗。")

        tk.Label(top, text="模型回复：").pack(anchor="w", pady=(8,0))
        self.txt_out = tk.Text(top, height=12, state="disabled", wrap="word"); self.txt_out.pack(fill="both", expand=True)

        bottom = tk.Frame(self, pady=8); bottom.pack(fill="x")
        self.btn_send = tk.Button(bottom, text="发送", width=10, command=self.on_send); self.btn_send.pack(side="right", padx=6)
        tk.Button(bottom, text="关闭", width=10, command=self.destroy).pack(side="right", padx=6)

    def on_send(self):
        prompt = self.txt_in.get("1.0","end").strip()
        if not prompt: messagebox.showwarning("提示","请输入内容"); return
        self.btn_send.config(state="disabled"); self._append_out("[正在请求模型...]\n")
        threading.Thread(target=self._call_api_safe, args=(prompt,), daemon=True).start()

    def _call_api_safe(self, prompt: str):
        try:
            text = call_model(self.provider, prompt)
        except Exception as e:
            text = f"[错误]\n{e}\n{traceback.format_exc(limit=1)}"
        self.after(0, lambda: (self._set_out(text), self.btn_send.config(state="normal")))

    def _set_out(self, content: str):
        self.txt_out.config(state="normal"); self.txt_out.delete("1.0","end"); self.txt_out.insert("1.0", content); self.txt_out.config(state="disabled")

    def _append_out(self, content: str):
        self.txt_out.config(state="normal"); self.txt_out.insert("end", content); self.txt_out.see("end"); self.txt_out.config(state="disabled")

# ===================== 对话框：批量分析（Qwen） =====================
class BatchAnalyzeDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("批量分析（Qwen 皮肤图像）")
        self.geometry("780x560"); self.resizable(True, True)

        frm = tk.Frame(self, padx=8, pady=8); frm.pack(fill="both", expand=True)
        tk.Label(frm, text="配置文件：config.json（需与程序同目录或搜索路径可见）").pack(anchor="w")
        self.txt_log = tk.Text(frm, height=24, wrap="word"); self.txt_log.pack(fill="both", expand=True, pady=(6,6))

        btns = tk.Frame(frm); btns.pack(fill="x")
        self.btn_run = tk.Button(btns, text="开始分析", command=self.run_batch); self.btn_run.pack(side="right", padx=6)
        tk.Button(btns, text="关闭", command=self.destroy).pack(side="right", padx=6)

    def log(self, s: str):
        self.txt_log.insert("end", s + "\n"); self.txt_log.see("end")

    def run_batch(self):
        self.btn_run.config(state="disabled")
        self.log("🚀 准备开始：读取 config.json 并启动多进程分析...")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            # 在子线程里运行你的流程；进程池会在内部创建，不会卡住界面
            multiprocessing.freeze_support()
            results, patient_df = analyze_skin_images_multimodal("config.json")
            if results:
                summary_path = create_multimodal_summary(results, patient_df)
            else:
                summary_path = None
        except Exception as e:
            err = f"[错误] 运行失败：{e}"
            self.after(0, lambda: (self.log(err), self.btn_run.config(state="normal")))
            return

        def finish():
            self.log("✅ 分析完成。")
            if summary_path and os.path.exists(summary_path):
                self.log(f"📋 汇总报告：{summary_path}")
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.log("\n=== 汇总报告内容预览 ===\n" + content)
                except Exception as e:
                    self.log(f"[提示] 无法读取报告内容：{e}")
            self.btn_run.config(state="normal")

        self.after(0, finish)

# ===================== 入口 =====================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = App(image_rel_path="assets/picture.jpg")
    app.mainloop()
