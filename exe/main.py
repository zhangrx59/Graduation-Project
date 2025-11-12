import os, sys, json, hashlib
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# =============== 路径工具（PyInstaller 资源只读；用户数据放本地可写目录） ===============
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

def app_data_dir():
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "ShowImageApp")
    os.makedirs(d, exist_ok=True)
    return d

USERS_FILE = os.path.join(app_data_dir(), "users.json")

# =============== 简单密码哈希（避免明文存储） ===============
def _hash(pwd: str) -> str:
    # 简单 sha256；生产可加盐/算法升级
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        # 首次运行创建默认账户
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

# =============== 主应用（登录 <-> 图片） ===============
class App(tk.Tk):
    def __init__(self, image_rel_path):
        super().__init__()
        self.title("登录示例")
        self.geometry("900x600")
        self.minsize(480, 320)

        self.image_rel_path = image_rel_path
        self.users = load_users()  # {'username': 'sha256(password)'}
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

    # ---- 注册窗口 ----
    def open_register(self):
        RegisterWindow(self, on_created=self._on_user_created, users_provider=lambda: self.users)

    def _on_user_created(self, username: str, password: str):
        u = username.strip()
        if not u:
            messagebox.showerror("注册失败", "用户名不能为空")
            return
        if u in self.users:
            messagebox.showerror("注册失败", "该用户名已存在")
            return
        if len(password) < 6:
            messagebox.showerror("注册失败", "密码长度至少 6 位")
            return
        self.users[u] = _hash(password)
        save_users(self.users)
        messagebox.showinfo("成功", f"用户 '{u}' 注册成功，可使用新账户登录。")

# =============== 登录界面 ===============
class LoginFrame(tk.Frame):
    def __init__(self, master: App, on_success, on_register, users_provider):
        super().__init__(master, padx=16, pady=16)
        self.on_success = on_success
        self.on_register = on_register
        self.users_provider = users_provider

        # 可选 logo
        logo_path = resource_path("assets/picture.jpg")
        self.tk_logo = None
        if os.path.exists(logo_path):
            try:
                _img = Image.open(logo_path)
                _img.thumbnail((260, 180))
                self.tk_logo = ImageTk.PhotoImage(_img)
                tk.Label(self, image=self.tk_logo).grid(row=0, column=0, columnspan=2, pady=(0, 10))
            except Exception:
                pass

        tk.Label(self, text="用户名").grid(row=1, column=0, sticky="e", pady=4, padx=(0, 6))
        self.ent_user = tk.Entry(self, width=24)
        self.ent_user.grid(row=1, column=1, pady=4)
        self.ent_user.insert(0, "admin")

        tk.Label(self, text="密码").grid(row=2, column=0, sticky="e", pady=4, padx=(0, 6))
        self.var_pwd = tk.StringVar()
        self.ent_pwd = tk.Entry(self, textvariable=self.var_pwd, width=24, show="•")
        self.ent_pwd.grid(row=2, column=1, pady=4)
        self.ent_pwd.insert(0, "123456")

        self.var_show = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="显示密码", variable=self.var_show,
                       command=lambda: self.ent_pwd.config(show="" if self.var_show.get() else "•")
        ).grid(row=3, column=1, sticky="w", pady=(0, 6))

        # 按钮区：登录 / 注册
        btn_area = tk.Frame(self)
        btn_area.grid(row=4, column=0, columnspan=2, pady=8)
        tk.Button(btn_area, text="登录", width=12, command=self.try_login).pack(side="left", padx=6)
        tk.Button(btn_area, text="注册", width=12, command=self.on_register).pack(side="left", padx=6)

        self.bind_all("<Return>", lambda e: self.try_login())
        for c in range(2):
            self.grid_columnconfigure(c, weight=1)

    def try_login(self):
        user = self.ent_user.get().strip()
        pwd  = self.ent_pwd.get()
        users = self.users_provider()
        if user in users and users[user] == _hash(pwd):
            self.on_success(user)
        else:
            messagebox.showerror("登录失败", "用户名或密码错误")

# =============== 注册窗口（Toplevel） ===============
class RegisterWindow(tk.Toplevel):
    def __init__(self, master: App, on_created, users_provider):
        super().__init__(master)
        self.title("注册")
        self.resizable(False, False)
        self.on_created = on_created
        self.users_provider = users_provider

        frm = tk.Frame(self, padx=16, pady=16)
        frm.grid(row=0, column=0)

        tk.Label(frm, text="用户名").grid(row=0, column=0, sticky="e", pady=4, padx=(0, 6))
        self.ent_user = tk.Entry(frm, width=24)
        self.ent_user.grid(row=0, column=1, pady=4)

        tk.Label(frm, text="密码").grid(row=1, column=0, sticky="e", pady=4, padx=(0, 6))
        self.ent_pwd = tk.Entry(frm, width=24, show="•")
        self.ent_pwd.grid(row=1, column=1, pady=4)

        tk.Label(frm, text="确认密码").grid(row=2, column=0, sticky="e", pady=4, padx=(0, 6))
        self.ent_pwd2 = tk.Entry(frm, width=24, show="•")
        self.ent_pwd2.grid(row=2, column=1, pady=4)

        tk.Button(frm, text="创建账户", width=20, command=self.create_user).grid(row=3, column=0, columnspan=2, pady=10)
        self.bind("<Return>", lambda e: self.create_user())

        self.after(50, self._center)

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def create_user(self):
        user = self.ent_user.get().strip()
        pwd1 = self.ent_pwd.get()
        pwd2 = self.ent_pwd2.get()
        users = self.users_provider()

        if not user:
            messagebox.showerror("错误", "用户名不能为空"); return
        if user in users:
            messagebox.showerror("错误", "该用户名已存在"); return
        if len(pwd1) < 6:
            messagebox.showerror("错误", "密码长度至少 6 位"); return
        if pwd1 != pwd2:
            messagebox.showerror("错误", "两次输入的密码不一致"); return

        # 回调到主程序写入
        self.on_created(user, pwd1)
        self.destroy()

# =============== 图片页（含退出登录/退出程序） ===============
class ImageFrame(tk.Frame):
    def __init__(self, master, image_rel_path, on_logout, on_exit):
        super().__init__(master)
        self.on_logout = on_logout
        self.on_exit = on_exit

        topbar = tk.Frame(self, bg="#444444", height=40)
        topbar.pack(fill="x", side="top")
        tk.Button(topbar, text="退出登录", command=self.logout).pack(side="right", padx=10, pady=5)
        tk.Button(topbar, text="退出程序", command=self.exit_program).pack(side="right", padx=10, pady=5)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#222222")
        self.canvas.pack(fill="both", expand=True)

        path = resource_path(image_rel_path)
        self.original = Image.open(path).convert("RGBA")
        self.tk_img = None

    def render(self):
        if not hasattr(self, "original"):
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.original.size
        scale = min(cw / iw, ch / ih)
        new_w = max(int(iw * scale), 1)
        new_h = max(int(ih * scale), 1)
        img_resized = self.original.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

    def logout(self):
        if messagebox.askyesno("确认", "确定要退出登录吗？"):
            self.on_logout()

    def exit_program(self):
        if messagebox.askyesno("确认", "确定要退出程序吗？"):
            self.on_exit()

# =============== 入口 ===============
if __name__ == "__main__":
    app = App(image_rel_path="assets/picture.jpg")
    app.mainloop()
