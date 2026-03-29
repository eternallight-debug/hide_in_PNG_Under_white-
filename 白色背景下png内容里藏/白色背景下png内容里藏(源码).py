import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import sys

# ------------------- 核心配置 & 计算函数 -------------------
WHITE_THRESHOLD_DEFAULT = 245
TARGET_ALPHA_DEFAULT = 0.5

def calculate_optimal_alpha_and_color_vectorized(original_rgb, target_alpha=0.5, white_threshold=245):
    """向量化计算：越白越透明的渐变逻辑"""
    whiteness = np.mean(original_rgb / 255.0, axis=1)
    whiteness_threshold = white_threshold / 255.0
    
    is_white = np.all(original_rgb >= white_threshold, axis=1)
    high_white = (whiteness >= whiteness_threshold) & (~is_white)
    normal_pixel = ~is_white & ~high_white
    
    alpha_low = (255 - original_rgb) / 255
    alpha_high = original_rgb / 255
    alpha_min = np.max(np.stack([alpha_low, alpha_high], axis=2), axis=(1,2))
    alpha_min = np.clip(alpha_min, 0.0, 1.0)
    
    final_alpha = np.zeros_like(whiteness)
    final_alpha[is_white] = 0.0
    
    if np.any(high_white):
        alpha_linear = target_alpha - (whiteness[high_white] - whiteness_threshold) / (1.0 - whiteness_threshold) * target_alpha
        final_alpha[high_white] = np.clip(alpha_linear, 0.0, target_alpha)
    
    final_alpha[normal_pixel] = np.where(alpha_min[normal_pixel] <= target_alpha, target_alpha, alpha_min[normal_pixel])
    final_alpha = np.clip(final_alpha, 0.01, 1.0)
    
    final_alpha_expanded = final_alpha[:, np.newaxis]
    target_rgb = (original_rgb - (1 - final_alpha_expanded) * 255) / final_alpha_expanded
    target_rgb[is_white] = 255
    target_rgb = np.clip(target_rgb, 0, 255).astype(np.uint8)
    
    target_bgr = target_rgb[:, [2, 1, 0]]
    return final_alpha, target_bgr

def process_images(path_A, path_B, output_path, white_threshold, target_alpha):
    """图片处理主函数 - 终极兼容修复版"""
    try:
        # 1. 路径规范化，兼容Windows系统
        path_A = os.path.normpath(path_A)
        path_B = os.path.normpath(path_B)
        output_path = os.path.normpath(output_path)

        # 2. 读取图片 - 二进制方式，彻底解决路径编码/读取失败问题
        with open(path_A, 'rb') as f:
            bytes_A = bytearray(f.read())
            A = cv2.imdecode(np.array(bytes_A), cv2.IMREAD_COLOR)
        with open(path_B, 'rb') as f:
            bytes_B = bytearray(f.read())
            B = cv2.imdecode(np.array(bytes_B), cv2.IMREAD_GRAYSCALE)

        # 3. 图片读取校验
        if A is None:
            raise Exception("前景图(A)读取失败，文件可能损坏、格式不支持或路径非法。")
        if B is None:
            raise Exception("蒙版图(B)读取失败，文件可能损坏、格式不支持或路径非法。")

        # 4. 尺寸对齐
        h, w = A.shape[:2]
        if B.shape[:2] != (h, w):
            B = cv2.resize(B, (w, h), interpolation=cv2.INTER_NEAREST)

        # 5. 核心像素处理
        result = np.zeros((h, w, 4), dtype=np.uint8)
        mask_white = (B != 0)
        mask_black = (B == 0)
        
        # 白色蒙版区域：保留原图，完全不透明
        result[mask_white, :3] = A[mask_white]
        result[mask_white, 3] = 255

        # 黑色蒙版区域：渐变透明处理
        if np.any(mask_black):
            original_pixels_rgb = A[mask_black][:, [2, 1, 0]]
            final_alpha, target_bgr = calculate_optimal_alpha_and_color_vectorized(
                original_pixels_rgb, target_alpha, white_threshold
            )
            result[mask_black, :3] = target_bgr
            result[mask_black, 3] = (final_alpha * 255).astype(np.uint8)

        # 6. 【核心修复】保存前自动创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 7. 【核心修复】二进制方式保存，彻底解决写入权限/路径问题
        ret, img_encode = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ret:
            raise Exception("图片编码失败，无法生成输出文件。")
        
        with open(output_path, 'wb') as f:
            f.write(img_encode)

        # 8. 校验是否保存成功
        if not os.path.exists(output_path):
            raise Exception("文件写入失败，请检查：\n1. 输出目录是否有写入权限\n2. 输出文件是否被其他程序占用\n3. 路径是否包含特殊字符/中文空格")

        return True, f"✅ 处理完成！\n📁 结果保存至：{output_path}"
        
    except PermissionError:
        return False, "❌ 权限不足：请关闭占用该文件的程序，或更换输出目录（建议保存到桌面）"
    except FileNotFoundError:
        return False, "❌ 文件不存在：请检查输入图片路径是否正确，输出目录是否合法"
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        line_num = tb.tb_lineno
        return False, f"❌ 异常发生在行 {line_num}：\n{str(e)}"

# ------------------- UI界面 -------------------
class ImageProcessorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("白色背景下PNG里内容藏")
        self.root.geometry("780x450")
        self.root.resizable(False, False)
        
        # 变量初始化
        self.path_A = tk.StringVar()
        self.path_B = tk.StringVar()
        self.output_path = tk.StringVar(value="result.png")
        self.white_threshold = tk.IntVar(value=WHITE_THRESHOLD_DEFAULT)
        self.target_alpha = tk.DoubleVar(value=round(TARGET_ALPHA_DEFAULT, 2))
        
        # 透明度精度约束
        self.target_alpha.trace('w', self._round_alpha)
        self._create_widgets()
    
    def _round_alpha(self, *args):
        """将透明度值四舍五入保留2位小数"""
        try:
            current_val = self.target_alpha.get()
            rounded_val = round(current_val, 2)
            if current_val != rounded_val:
                self.target_alpha.set(rounded_val)
        except:
            pass
    
    def _create_widgets(self):
        # 1. 文件选择区域
        frame1 = ttk.LabelFrame(self.root, text="文件选择")
        frame1.pack(padx=15, pady=10, fill=tk.X)
        
        # 前景图A
        ttk.Label(frame1, text="前景图(A)：").grid(row=0, column=0, padx=5, pady=6, sticky=tk.W)
        ttk.Entry(frame1, textvariable=self.path_A, width=55).grid(row=0, column=1, padx=5, pady=6)
        ttk.Button(frame1, text="选择文件", command=self._select_A, width=12).grid(row=0, column=2, padx=8, pady=6)
        
        # 蒙版图B
        ttk.Label(frame1, text="蒙版图(B)：").grid(row=1, column=0, padx=5, pady=6, sticky=tk.W)
        ttk.Entry(frame1, textvariable=self.path_B, width=55).grid(row=1, column=1, padx=5, pady=6)
        ttk.Button(frame1, text="选择文件", command=self._select_B, width=12).grid(row=1, column=2, padx=8, pady=6)
        
        # 输出路径
        ttk.Label(frame1, text="输出路径：").grid(row=2, column=0, padx=5, pady=6, sticky=tk.W)
        ttk.Entry(frame1, textvariable=self.output_path, width=55).grid(row=2, column=1, padx=5, pady=6)
        ttk.Button(frame1, text="选择保存", command=self._select_output, width=12).grid(row=2, column=2, padx=8, pady=6)
        
        # 2. 参数调整区域
        frame2 = ttk.LabelFrame(self.root, text="渐变参数调整")
        frame2.pack(padx=15, pady=5, fill=tk.X)
        
        # 纯白阈值
        ttk.Label(frame2, text="纯白阈值(200-255)：").grid(row=0, column=0, padx=5, pady=8, sticky=tk.W)
        ttk.Scale(frame2, from_=200, to=255, variable=self.white_threshold, orient=tk.HORIZONTAL, length=380).grid(row=0, column=1, padx=5, pady=8)
        ttk.Label(frame2, textvariable=self.white_threshold, width=5).grid(row=0, column=2, padx=5, pady=8)
        
        # 基础透明度
        ttk.Label(frame2, text="基础透明度(0.0-1.0)：").grid(row=1, column=0, padx=5, pady=8, sticky=tk.W)
        ttk.Scale(frame2, from_=0.0, to=1.0, variable=self.target_alpha, orient=tk.HORIZONTAL, length=380).grid(row=1, column=1, padx=5, pady=8)
        ttk.Label(frame2, textvariable=self.target_alpha, width=5).grid(row=1, column=2, padx=5, pady=8)
        
        # 3. 执行按钮
        frame3 = ttk.Frame(self.root)
        frame3.pack(padx=15, pady=12)
        ttk.Button(frame3, text="🚀 开始处理", command=self._process, width=20).grid(row=0, column=0, padx=10)
        
        # 4. 日志输出区域
        frame4 = ttk.LabelFrame(self.root, text="处理日志")
        frame4.pack(padx=15, pady=5, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(frame4, height=7, width=85, state=tk.NORMAL)
        self.log_text.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
    
    def _select_A(self):
        path = filedialog.askopenfilename(
            title="选择前景图(A)",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if path:
            self.path_A.set(path)
            # 自动填充输出路径到同目录
            dir_name = os.path.dirname(path)
            default_out = os.path.join(dir_name, "result.png")
            self.output_path.set(default_out)
            self._log(f"已选择前景图：{os.path.basename(path)}")
    
    def _select_B(self):
        path = filedialog.askopenfilename(
            title="选择蒙版图(B)",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if path:
            self.path_B.set(path)
            self._log(f"已选择蒙版图：{os.path.basename(path)}")
    
    def _select_output(self):
        path = filedialog.asksaveasfilename(
            title="选择保存位置",
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")]
        )
        if path:
            self.output_path.set(path)
            self._log(f"已设置输出路径：{path}")
    
    def _log(self, msg):
        """日志输出"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def _process(self):
        """执行处理逻辑"""
        # 输入校验
        path_a = self.path_A.get().strip()
        path_b = self.path_B.get().strip()
        output = self.output_path.get().strip()

        if not path_a or not os.path.exists(path_a):
            messagebox.showerror("输入错误", "请选择有效的前景图(A)！")
            return
        if not path_b or not os.path.exists(path_b):
            messagebox.showerror("输入错误", "请选择有效的蒙版图(B)！")
            return
        if not output:
            messagebox.showerror("输入错误", "请设置有效的输出路径！")
            return
        
        # 清空日志
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log("🔄 开始处理...")
        self._log(f"前景图路径：{path_a}")
        self._log(f"蒙版图路径：{path_b}")
        self._log(f"输出路径：{output}")
        self._log(f"纯白阈值：{self.white_threshold.get()}")
        self._log(f"基础透明度：{self.target_alpha.get()}")
        self._log("------------------------")
        
        # 执行处理
        success, msg = process_images(
            path_a,
            path_b,
            output,
            self.white_threshold.get(),
            self.target_alpha.get()
        )
        
        # 输出结果
        self._log(msg)
        if success:
            self._log("✅ 全部任务处理完成！")
            messagebox.showinfo("处理成功", "图片处理完成！\n已保存到指定路径")
        else:
            messagebox.showerror("处理失败", msg)

# ------------------- 主程序入口 -------------------
if __name__ == "__main__":
    # 适配Windows高DPI
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    root = tk.Tk()
    app = ImageProcessorUI(root)
    root.mainloop()
