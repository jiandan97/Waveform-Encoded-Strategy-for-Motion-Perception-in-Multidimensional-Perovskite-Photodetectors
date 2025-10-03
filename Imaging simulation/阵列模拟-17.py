# -*- coding: utf-8 -*-
"""
阵列模拟—稳态响应—带 mask & 透明区打孔（多事件版：每次不透明进入都触发一次完整波形）
根据原始波形的 S2、S3、S4 三段映射：
- S2: 原始波形 0.000716s→0.001453s
- S3: 由每次接触到的完全不透明光源实际长度减去 PIX_W，再除以速度得到时长，
      线性衔接 S2 末端与 S4 起始电流
- S4: 原始波形 0.003219s→0.003956s
最终只打印**全局**最短接触长度与 PIX_W 的比较结果。
"""
import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import tkinter as tk
from tkinter import filedialog

import matplotlib.image as mpimg
from PIL import Image

# —— 中文 & 负号 ——
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

# —— 配置参数 ——
TIME_SCALE       = 0.03        # 视频中 1 秒 对应 仿真秒数
VIDEO_FPS        = 30          # 动画帧率
VIDEO_DPI        = 400         # MP4 输出 DPI
VIEW_MARGIN_FACTOR = 8.0      # 视图边距因子

# 光源运动参数
START_X, START_Y = -850.0, -80.0  # 光源初始位置 (mm)
MOVE_DIST        = 1150.0       # 光源移动总距离 (mm)
ANGLE_DEG        = 0.0          # 光源运动角度 (deg)
SPEED            = 7330.4       # 光源运动速度 (mm/s)
OFFSET_WT        = 0.000716     # 波形时间偏移 (s)

# 图片缩放参数：在物理尺寸基础上的额外比例
IMG_SCALE_FACTOR = 15.0         # 1.0 保持原始物理尺寸

# 像素阵列参数
N_COL, N_ROW     = 100, 100     # 像素阵列列数、行数
PIX_W, PIX_H     = 5.4, 3.25    # 像素总宽度、高度 (mm)
LEFT_W           = 0.5          # 像素左侧部分宽度 (mm)
PIX_SPACING      = 0.5          # 像素间距 (mm)

# 遮罩区域
MASK_X0, MASK_Y0       = -150.0, 130.0  # 遮罩区域左下角坐标 (mm)
MASK_WIDTH, MASK_HEIGHT = 300.0, 150.0  # 遮罩区域尺寸 (mm)

# GRID 输出尺寸
SUBPLOT_W_INCH   = 1.5         # 子图宽度 (inch)
SUBPLOT_H_INCH   = 1.5         # 子图高度 (inch)


def get_mm_per_px(img_path: str) -> Optional[float]:
    """
    从图像 DPI 元数据读取像素→毫米比例，否则返回 None
    """
    with Image.open(img_path) as im:
        info = im.info
        dpi = info.get('dpi')
        if isinstance(dpi, tuple) and dpi[0] > 0:
            return 25.4 / dpi[0]
    return None


def simulate_pixel_response(
    ts: np.ndarray,
    t_vals: np.ndarray,
    I_vals: np.ndarray,
    pix_left: float,
    pix_right: float,
    pix_bottom: float,
    pix_top: float,
    start_x: float,
    start_y: float,
    vx: float,
    vy: float,
    offset_wt: float,
    last_current: float,
    alpha_channel: np.ndarray,
    w_px: int,
    h_px: int,
    insertion_offset: float = 0.000737
) -> Tuple[np.ndarray, Optional[float]]:
    """
    模拟单个像素响应并返回电流波形与最短接触长度
    """
    N = len(ts)
    I_out = np.zeros(N)
    mask = np.zeros(N, dtype=bool)

    for i, t in enumerate(ts):
        x_l = start_x + vx * t
        y_l = start_y + vy * t

        # 计算对应图像像素坐标
        u0 = int(
            ((pix_left - x_l) /
             (w_px * mm_per_px * IMG_SCALE_FACTOR))
            * w_px
        )
        v0 = int(
            ((pix_bottom - y_l) /
             (h_px * mm_per_px * IMG_SCALE_FACTOR))
            * h_px
        )

        u0, u1 = max(0, u0), min(w_px, u0 + 1)
        v0, v1 = max(0, v0), min(h_px, v0 + 1)

        # 如果对应像素 alpha >= 1，则认为有接触
        if u0 < u1 and v0 < v1 and np.any(alpha_ch[v0:v1, u0:u1] >= 1.0):
            mask[i] = True

    event_starts = np.where(
        mask & ~np.concatenate(([False], mask[:-1]))
    )[0]
    speed = np.hypot(vx, vy)
    s2 = insertion_offset
    s4s = 0.003219
    s4 = 0.003956 - s4s

    lengths = []
    for idx0 in event_starts:
        # 找到事件结束的索引
        if np.any(~mask[idx0:]):
            end_idx = idx0 + np.argmax(~mask[idx0:])
        else:
            end_idx = N

        length = speed * (ts[end_idx - 1] - ts[idx0])
        lengths.append(length)

        # 计算平台时长
        excess = max(0, length - PIX_W) / speed
        I2 = np.interp(offset_wt + s2, t_vals, I_vals)
        I4 = np.interp(s4s, t_vals, I_vals)
        total_t = s2 + excess + s4

        # 填充 I_out
        for k, dt in enumerate(ts[idx0:] - ts[idx0]):
            if dt >= total_t:
                break
            j = idx0 + k
            if dt < s2:
                I_out[j] = np.interp(
                    offset_wt + dt, t_vals, I_vals,
                    right=last_current
                )
            elif dt < s2 + excess:
                frac = (dt - s2) / excess if excess > 0 else 1
                I_out[j] = I2 + frac * (I4 - I2)
            else:
                I_out[j] = np.interp(
                    s4s + (dt - s2 - excess),
                    t_vals, I_vals,
                    right=last_current
                )

    return I_out, (min(lengths) if lengths else None)


# —— 读取波形 CSV ——
root = tk.Tk()
root.withdraw()
csv_path = filedialog.askopenfilename(
    title="选择电流波形 CSV",
    filetypes=[("CSV", "*.csv")]
)
root.destroy()
if not csv_path:
    raise SystemExit("未选择 CSV 文件")

df = pd.read_csv(csv_path)
t_vals = df['Time_s'].to_numpy()
I_vals = df['Current_A'].to_numpy()
offset_cur = I_vals[np.abs(t_vals - OFFSET_WT).argmin()]
last_cur = I_vals[-1]

# —— 读取图片并计算物理尺寸 ——
root = tk.Tk()
root.withdraw()
img_path = filedialog.askopenfilename(
    title="选择光源图片",
    filetypes=[("Image", "*.png;*.jpg;*.bmp")]
)
root.destroy()
light_img = mpimg.imread(img_path)
h_px, w_px = light_img.shape[:2]
mm_per_px = get_mm_per_px(img_path)
if mm_per_px is None:
    print("未检出 DPI，程序退出。")
    sys.exit(1)

# 应用缩放倍数后的实际尺寸
img_w_mm = w_px * mm_per_px * IMG_SCALE_FACTOR
img_h_mm = h_px * mm_per_px * IMG_SCALE_FACTOR

# 如果图像没有 alpha 通道，则创建全 1 的 alpha
if light_img.ndim == 3 and light_img.shape[2] == 3:
    alpha = np.ones((h_px, w_px))
    light_img = np.dstack([light_img, alpha])

alpha_ch = np.flipud(light_img[:, :, 3])

# —— 时间与运动 ——
dt = 1e-5
sim_t_end = MOVE_DIST / SPEED
ts_all = np.linspace(
    0, sim_t_end, int(sim_t_end / dt) + 1
)
theta = np.deg2rad(ANGLE_DEG)
vx = SPEED * np.cos(theta)
vy = SPEED * np.sin(theta)
frame_idxs = list(
    range(
        0,
        len(ts_all),
        max(1, int((TIME_SCALE / VIDEO_FPS) / dt))
    )
)

# —— 像素阵列布局 ——
center = (800 / 2) / 100
tw = N_COL * PIX_W + (N_COL - 1) * PIX_SPACING
first_x = center - tw / 2
pix_lefts = [
    first_x + i * (PIX_W + PIX_SPACING)
    for i in range(N_COL)
]
pix_tops = [
    j * (PIX_H + PIX_SPACING)
    for j in range(N_ROW)
]

# —— 预览动画 ——
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax.set_aspect('equal')

mx = (PIX_W + PIX_SPACING) * VIEW_MARGIN_FACTOR
my = (PIX_H + PIX_SPACING) * VIEW_MARGIN_FACTOR
ax.set_xlim(first_x - mx, first_x + tw + mx)
ax.set_ylim(-my, N_ROW * (PIX_H + PIX_SPACING) - PIX_SPACING + my)

# 绘制像素阵列
for x0 in pix_lefts:
    for y0 in pix_tops:
        ax.add_patch(
            Rectangle((x0, y0), LEFT_W, PIX_H,
                      facecolor='red', alpha=0.8)
        )
        ax.add_patch(
            Rectangle((x0 + LEFT_W, y0),
                      PIX_W - LEFT_W, PIX_H,
                      facecolor='blue', alpha=0.8)
        )

# 绘制遮罩区域
ax.add_patch(
    Rectangle(
        (MASK_X0, MASK_Y0),
        MASK_WIDTH, MASK_HEIGHT,
        facecolor='black', edgecolor='none',
        alpha=1.0
    )
)

# 显示光源图片
light_im = ax.imshow(
    light_img,
    extent=(START_X, START_X + img_w_mm,
            START_Y, START_Y + img_h_mm),
    origin='upper',
    aspect='equal',
    alpha=1,
    zorder=2
)
ax.set_title("Pattern motion schematic")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")

time_text = ax.text(
    0.02, 0.95, '', color="red", 
    transform=ax.transAxes
)


def update(i):
    t = ts_all[i]
    x0 = START_X + vx * t
    y0 = START_Y + vy * t

    light_im.set_extent(
        (x0, x0 + img_w_mm, y0, y0 + img_h_mm)
    )
    time_text.set_text(f"t={t:.6f}s")
    return light_im, time_text


anim = FuncAnimation(
    fig, update, frames=frame_idxs,
    interval=1000 / VIDEO_FPS, blit=False
)
plt.show()

# —— 输出 CSV/GRID/MP4 ——
choices = input(
    "生成哪些？CSV,GRID,MP4 或 ALL："
).upper().split(',')
want_csv = 'CSV' in choices or 'ALL' in choices
want_grid = 'GRID' in choices or 'ALL' in choices
want_mp4 = 'MP4' in choices or 'ALL' in choices

out_dir = os.getcwd()

if want_mp4:
    fig2, ax2 = plt.subplots(
        figsize=(8, 5), constrained_layout=True
    )
    ax2.set_aspect('equal')

    ax2.set_title("Pattern motion schematic")
    ax2.set_xlabel("X position (mm)")
    ax2.set_ylabel("Y position (mm)")

    ax2.set_xlim(first_x - mx, first_x + tw + mx)
    ax2.set_ylim(
        -my,
        N_ROW * (PIX_H + PIX_SPACING) - PIX_SPACING + my
    )

    # 绘制像素阵列
    for x0 in pix_lefts:
        for y0 in pix_tops:
            ax2.add_patch(
                Rectangle((x0, y0), LEFT_W, PIX_H,
                          facecolor='red',
                          alpha=0.8,
                          zorder=2)
            )
            ax2.add_patch(
                Rectangle((x0 + LEFT_W, y0),
                          PIX_W - LEFT_W, PIX_H,
                          facecolor='blue',
                          alpha=0.8,
                          zorder=2)
            )

    # 遮罩区
    ax2.add_patch(
        Rectangle(
            (MASK_X0, MASK_Y0),
            MASK_WIDTH, MASK_HEIGHT,
            facecolor='black',
            edgecolor='none',
            alpha=1.0,
            zorder=2
        )
    )

    lm2 = ax2.imshow(
        light_img,
        extent=(START_X, START_X + img_w_mm,
                START_Y, START_Y + img_h_mm),
        origin='upper',
        aspect='equal',
        alpha=1,
        zorder=3
    )
    tt2 = ax2.text(
        0.02, 0.95, '',
        transform=ax2.transAxes,
        zorder=4
    )
    canvas2 = FigureCanvas(fig2)
    writer = FFMpegWriter(fps=VIDEO_FPS)

    with writer.saving(
        fig2,
        os.path.join(out_dir, 'light_motion.mp4'),
        dpi=VIDEO_DPI
    ):
        for i in frame_idxs:
            t = ts_all[i]
            x0 = START_X + vx * t
            y0 = START_Y + vy * t

            lm2.remove()
            lm2 = ax2.imshow(
                light_img,
                extent=(x0, x0 + img_w_mm,
                        y0, y0 + img_h_mm),
                origin='upper',
                aspect='equal',
                alpha=1,
                zorder=3
            )
            tt2.set_text(f"t={t:.6f}s")
            canvas2.draw()
            writer.grab_frame()

    print(
        '已保存 MP4→',
        os.path.join(out_dir, 'light_motion.mp4')
    )
    plt.close(fig2)

if want_csv or want_grid:
    I_grid = np.zeros((N_ROW, N_COL, len(ts_all)))
    wave_dir = os.path.join(out_dir, 'pixel_waveforms')
    if want_csv:
        os.makedirs(wave_dir, exist_ok=True)
    
    all_l = []
    occluded_pixels = []
    for c, x0 in enumerate(pix_lefts):
        for r, y0 in enumerate(pix_tops):
            x1, y1 = x0 + PIX_W, y0 + PIX_H

            # 如果在遮罩区内，波形置零
            if (x0 < MASK_X0 + MASK_WIDTH and
                x1 > MASK_X0 and
                y0 < MASK_Y0 + MASK_HEIGHT and
                y1 > MASK_Y0):
                series, minl = np.zeros(len(ts_all)), None
            # 记录被遮挡的像素
                occluded_pixels.append({'row': r, 'col': c})
            else:
                series, minl = simulate_pixel_response(
                    ts_all, t_vals, I_vals,
                    x0, x1, y0, y1,
                    START_X, START_Y,
                    vx, vy,
                    OFFSET_WT,
                    last_cur,
                    alpha_ch,
                    w_px, h_px
                )

            I_grid[r, c] = series
            if minl is not None:
                all_l.append(minl)

            if want_csv:
                pd.DataFrame({
                    'Time_s': ts_all,
                    'Current_A': series
                }).to_csv(
                    os.path.join(
                        wave_dir,
                        f'pixel_{r}_{c}.csv'
                    ),
                    index=False
                )
    # 在这里将 occluded_pixels 列表写成 CSV
    if occluded_pixels:
        df_occl = pd.DataFrame(occluded_pixels)
        occl_path = os.path.join(out_dir, 'occluded_pixels.csv')
        df_occl.to_csv(occl_path, index=False)
        print(f'已保存被遮挡像素列表 → {occl_path}')
    else:
        print('没有检测到被遮挡的像素。')

    # 打印全局最短接触长度
    if all_l:
        gm = min(all_l)
        comp = '>' if gm > PIX_W else '≤'
        print(
            f"全局最短接触长度 {gm:.4f}mm "
            f"{comp} PIX_W({PIX_W}mm)"
        )

    if want_grid:
        fig_g, axes = plt.subplots(
            N_ROW, N_COL,
            figsize=(
                N_COL * SUBPLOT_W_INCH,
                N_ROW * SUBPLOT_H_INCH
            ),
            squeeze=False
        )
        axes = axes[::-1, :]
        for r in range(N_ROW):
            for c in range(N_COL):
                axg = axes[r, c]
                axg.plot(ts_all, I_grid[r, c], lw=0.8)
                axg.set_title(f'({r},{c})', fontsize=6)
                axg.tick_params(labelsize=4)
                axg.set_xlabel('Time(s)', fontsize=5)
                axg.set_ylabel('Current(A)', fontsize=5)

        fig_g.tight_layout()
        png = os.path.join(
            out_dir, 'pixel_waveforms_grid.png'
        )
        fig_g.savefig(png, dpi=200)
        plt.close(fig_g)
        print('已保存 GRID→', png)
