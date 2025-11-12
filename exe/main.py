import os, sys
import tkinter as tk
from PIL import Image, ImageTk

def resource_path(rel):
    # PyInstaller 支持：打包后从临时目录取资源
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

class ImageApp(tk.Tk):
    def __init__(self, image_rel_path):
        super().__init__()
        self.title("Show Image")

        # 允许窗口拉伸
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # 画布用来显示图片
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#222222")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # 载图（原图保留一份，缩放时基于原图重采样，避免越缩越糊）
        path = resource_path(image_rel_path)
        self.original = Image.open(path).convert("RGBA")

        # 首次渲染
        self._render()

        # 监听窗口大小变化，做等比缩放
        self.bind("<Configure>", self._on_resize)

        # 初始尺寸（可自行调整）
        self.geometry("900x600")

    def _on_resize(self, _event=None):
        # 只在尺寸变化时重绘
        self._render()

    def _render(self):
        if not hasattr(self, "original"):
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.original.size

        # 计算等比尺寸
        scale = min(cw / iw, ch / ih)
        new_w = max(int(iw * scale), 1)
        new_h = max(int(ih * scale), 1)

        # 抗锯齿缩放
        img_resized = self.original.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)

        self.canvas.delete("all")
        # 居中绘制
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_img)

if __name__ == "__main__":
    # 高 DPI 适配（Tk 8.6+ 在 Win11 基本OK，必要时可手动设置 scaling）
    # import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
    app = ImageApp("assets/picture.jpg")
    app.mainloop()
