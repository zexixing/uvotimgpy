import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from typing import Union, List, Dict, Any, Optional, Tuple
import math
import pandas as pd
import plotly.express as px


class MaskInspector:
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        """Initialize ImageInspector with an image and its mask
        
        Parameters
        ----------
        image : np.ndarray
            The original image array
        mask : np.ndarray
            Boolean mask array with same shape as image, True indicates masked pixels
        """
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape")
        self.image = image
        self.mask = mask
    
    def show_masked(self, figsize: Tuple[int, int] = (10, 8),
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   cmap: str = 'viridis',
                   title: str = 'Masked Pixels') -> None:
        """Display only the masked pixels
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling
        cmap : str
            Colormap name
        title : str
            Plot title
        """
        # Create an image containing only masked pixels
        masked_img = np.copy(self.image)
        masked_img[~self.mask] = np.nan  # Set unmasked pixels to nan
        
        plt.figure(figsize=figsize)
        plt.imshow(masked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(label='Pixel Value')
        plt.title(title)
        plt.show()
    
    def show_unmasked(self, figsize: Tuple[int, int] = (10, 8),
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      xlim: Optional[Tuple[float, float]] = None,
                      ylim: Optional[Tuple[float, float]] = None,
                      cmap: str = 'viridis',
                      title: str = 'Unmasked Pixels') -> None:
        """Display only the unmasked pixels
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling
        cmap : str
            Colormap name
        title : str
            Plot title
        """
        # Create an image containing only unmasked pixels
        unmasked_img = np.copy(self.image)
        unmasked_img[self.mask] = np.nan  # Set masked pixels to nan
        
        plt.figure(figsize=figsize)
        plt.imshow(unmasked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(label='Pixel Value')
        plt.title(title)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.show()
    
    def show_comparison(self, figsize: Tuple[int, int] = (15, 5),
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       cmap: str = 'viridis') -> None:
        """Display masked and unmasked pixels side by side
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height)
        vmin, vmax : float, optional
            Minimum and maximum values for color scaling
        cmap : str
            Colormap name
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # left panel: original image
        im1 = ax1.imshow(self.image, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(im1, ax=ax1, label='Pixel Value')
        ax1.set_title('Original Image')
        
        # middle panel: masked pixels
        masked_img = np.copy(self.image)
        masked_img[~self.mask] = np.nan
        im2 = ax2.imshow(masked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(im2, ax=ax2, label='Pixel Value')
        ax2.set_title('Masked Pixels')
        
        # Right panel: unmasked pixels
        unmasked_img = np.copy(self.image)
        unmasked_img[self.mask] = np.nan
        im3 = ax3.imshow(unmasked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(im3, ax=ax3, label='Pixel Value')
        ax3.set_title('Unmasked Pixels')
        
        plt.tight_layout()
        plt.show(block=True)


def draw_direction_compass(ax,
                           directions={'N': 0, 'E': 90, 'v': 240, '☉': 200},
                           colors='white',
                           position=(0.9, 0.9),
                           arrow_length=0.08,
                           arrow_width=0.5,
                           headwidth=4,
                           headlength=3,
                           text_offset=0.02,
                           fontsize=10):
    """
    在指定的 matplotlib 轴上绘制自定义方向的指南针（角度逆时针），每个方向可自定义颜色。
    
    参数：
    - ax: matplotlib 的轴对象
    - directions: dict，键为方向标签，值为角度（从N方向起，逆时针度数）
    - colors: 单一颜色字符串，或一个 dict，例如 {'N': 'red', 'E': 'green'} 表示每个方向的颜色
    - position: 指南针中心在图中位置（以 ax.transAxes 坐标表示，范围 0-1）
    - arrow_length: 箭头长度（以 Axes 坐标表示）
    - text_offset: 文本相对箭头终点的延伸距离（以 Axes 坐标表示）
    - fontsize: 文字大小

    返回：
    - ax: 修改后的 matplotlib 轴
    """
    x0, y0 = position

    # 统一颜色处理
    if isinstance(colors, str):
        # 所有方向统一颜色
        colors = {k: colors for k in directions.keys()}
    else:
        # 若提供部分颜色，未指定的用白色
        colors = {k: colors.get(k, 'white') for k in directions.keys()}

    for label, angle_deg in directions.items():
        angle_rad = np.deg2rad(angle_deg)
        dx = -np.sin(angle_rad)
        dy = np.cos(angle_rad)

        x1 = x0 + arrow_length * dx
        y1 = y0 + arrow_length * dy

        c = colors[label]

        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(facecolor=c, edgecolor=c, width=arrow_width, headwidth=headwidth, headlength=headlength),
                    xycoords='axes fraction')

        xt = x0 + (arrow_length + text_offset) * dx
        yt = y0 + (arrow_length + text_offset) * dy

        ax.text(xt, yt, label, color=c, fontsize=fontsize,
                ha='center', va='center', transform=ax.transAxes)

    return ax

def draw_scalebar(ax,
                  length=100,  # 以像素为单位
                  label_top='100 000 km',
                  label_bottom='45″',
                  position=(0.9, 0.9),
                  color='white',
                  linewidth=2,
                  text_offset=10,  # 文字偏移，以像素为单位
                  fontsize=10):
    """
    绘制比例尺，横线长度以像素为单位，位置以 Axes 坐标表示。
    
    参数：
    - ax: matplotlib Axes 对象
    - length: 横线长度（单位：像素）
    - label_top: 上方标签文本（可为 None）
    - label_bottom: 下方标签文本（可为 None）
    - position: 横线中心位置（以 Axes 坐标表示，0~1）
    - color: 颜色（用于线和文字）
    - linewidth: 线宽（单位：像素）
    - text_offset: 上下标签离横线的距离（单位：像素）
    - fontsize: 字体大小

    返回：
    - ax: 修改后的 matplotlib Axes 对象
    """
    # 取中心点（Axes → Data）
    x0_data, y0_data = ax.transAxes.transform(position)
    x0_data, y0_data = ax.transData.inverted().transform((x0_data, y0_data))

    # 横线起止 (data coords)
    x_start = x0_data - length / 2
    x_end   = x0_data + length / 2

    # 画横线
    ax.plot([x_start, x_end], [y0_data, y0_data],
            color=color, linewidth=linewidth)

    # 文本位置：在显示坐标中做偏移
    x0_disp, y0_disp = ax.transData.transform((x0_data, y0_data))
    y_top_disp = y0_disp + text_offset
    y_bottom_disp = y0_disp - text_offset
    _, y_top_data = ax.transData.inverted().transform((x0_disp, y_top_disp))
    _, y_bottom_data = ax.transData.inverted().transform((x0_disp, y_bottom_disp))

    if label_top:
        ax.text((x_start+x_end)/2, y_top_data, label_top,
                ha='center', va='bottom', color=color, fontsize=fontsize)
    if label_bottom:
        ax.text((x_start+x_end)/2, y_bottom_data, label_bottom,
                ha='center', va='top', color=color, fontsize=fontsize)

    return ax
    
def multi_show(image_list, max_cols=4, vrange: Union[None, Tuple[float, float], List[Tuple[float, float]]] = None, 
               xrange: Union[None, Tuple[float, float], List[Tuple[float, float]]] = None, 
               yrange: Union[None, Tuple[float, float], List[Tuple[float, float]]] = None,
               title_list: Union[None, List[str]] = None, 
               target_position_range: Union[None, Tuple[float, float], List[Tuple[float, float]]] = None,):
    """
    Display multiple images in a grid.

    Parameters
    ----------
    image_list : list
        List of images to display
    xrange: list
    yrange: list
    """
    n_images = len(image_list)
    n_cols = min(max_cols, n_images)
    n_rows = math.ceil(n_images / n_cols)

    # 创建和显示
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1: axes = [axes]
    if n_cols == 1: axes = [[ax] for ax in axes]

    if isinstance(xrange, Tuple):
        xrange_low = xrange[0]
        xrange_high = xrange[1]
        update_xrange = False
    elif isinstance(xrange, list) and isinstance(xrange[0], Tuple):
        if len(xrange) == len(image_list):
            update_xrange = True
        else:
            print('xrange length does not match image_list length')
    else:
        update_xrange = False

    if isinstance(yrange, Tuple):
        yrange_low = yrange[0]
        yrange_high = yrange[1]
        update_yrange = False
    elif isinstance(yrange, list) and isinstance(yrange[0], Tuple):
        if len(yrange) == len(image_list):
            update_yrange = True
        else:
            print('yrange length does not match image_list length')
    else:
        update_yrange = False
    
    if isinstance(vrange, Tuple):
        vrange_low = vrange[0]
        vrange_high = vrange[1]
        update_vrange = False
    elif isinstance(vrange, list) and isinstance(vrange[0], Tuple):
        if len(vrange) == len(image_list):
            update_vrange = True
        else:
            print('vrange length does not match image_list length')
    else:
        update_vrange = False

    if isinstance(target_position_range, Tuple):
        target_col = target_position_range[0]
        target_row = target_position_range[1]
        update_target_position_range = False
    elif isinstance(target_position_range, list) and isinstance(target_position_range[0], Tuple):
        if len(target_position_range) == len(image_list):
            update_target_position_range = True
        else:
            print('target_position_range length does not match image_list length')
    else:
        update_target_position_range = False

    for i, img in enumerate(image_list):
        row, col = i // n_cols, i % n_cols
        if update_xrange:
            xrange_low = xrange[i][0]
            xrange_high = xrange[i][1]
        if update_yrange:
            yrange_low = yrange[i][0]
            yrange_high = yrange[i][1]
        if update_vrange:
            vrange_low = vrange[i][0]
            vrange_high = vrange[i][1]
        if update_target_position_range:
            target_col = target_position_range[i][0]
            target_row = target_position_range[i][1]
        if vrange is None:
            axes[row][col].imshow(img, origin='lower')
        else:
            axes[row][col].imshow(img, vmin=vrange_low, vmax=vrange_high, origin='lower')
        axes[row][col].axis('off')
        if xrange is not None:
            axes[row][col].set_xlim(xrange_low, xrange_high)
        if yrange is not None:
            axes[row][col].set_ylim(yrange_low, yrange_high)
        if title_list is not None:
            axes[row][col].set_title(title_list[i])
        #axes[row][col].grid(color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        if target_position_range is not None:
            axes[row][col].plot(target_col, target_row, 'rx', markersize=5)

    # 隐藏多余子图
    for i in range(n_images, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row][col].set_visible(False)
    plt.tight_layout(pad=0.1)
    return fig, axes
    
def linear_stretch(img, p_low=2.0, p_high=98.0, mask=None, return_threshold=False):
    img = np.asarray(img, dtype=np.float32)

    # 直接忽略 NaN 计算百分位
    if mask is not None:
        lo, hi = np.nanpercentile(img[mask], [p_low, p_high])
    else:
        lo, hi = np.nanpercentile(img, [p_low, p_high])

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.zeros_like(img, dtype=np.float32)

    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo)
    if return_threshold:
        return out, lo, hi
    else:
        return out

def figs_to_gif(figs, save_path, duration=500):
    """
    Save figures as GIF.

    Parameters
    ----------
    figs : list
        List of figures
    outfile : str or pathlib.Path
        Output file path
    duration : int
        Duration of each frame in milliseconds
    """
    frames = []
    for fig in figs:
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        frames.append(Image.fromarray(buf))

    # 存 GIF：每帧 1000 毫秒（1s）
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,   # 毫秒！不是秒
        loop=0,          # 无限循环
        disposal=2       # 避免残影
    )

def plot_obs_timeline(
    csv_path,
    start=None,              # e.g. "2025-01-01 00:00:00" / pd.Timestamp(...)
    end=None,                # e.g. "2025-01-31 23:59:59"
    html_out=None,
    color_map = None,
    y_map = None,
    time_keyword='MIDTIME',
    obs_id_keyword='OBSID',
    filter_keyword='FILTER',
    datatype_keyword='DATATYPE',
    symbol_map = None,
    ):
    # 关键：OBSID/obsid 用字符串读，保留前导 0
    df = pd.read_csv(csv_path, dtype={obs_id_keyword: "string", obs_id_keyword.lower(): "string"})

    # 兼容列名大小写：优先用 OBSID，其次 obsid
    obs_col = obs_id_keyword if obs_id_keyword in df.columns else obs_id_keyword.lower()
    df[obs_col] = df[obs_col].astype("string")

    # MIDTIME 转 datetime
    df[time_keyword] = pd.to_datetime(df[time_keyword], errors="coerce", utc=False)
    df = df.dropna(subset=[time_keyword]).copy()

    # 时间范围：未提供就用表内最早/最晚
    start = pd.to_datetime(start) if start is not None else df[time_keyword].min()
    end   = pd.to_datetime(end)   if end   is not None else df[time_keyword].max()
    df = df[(df[time_keyword] >= start) & (df[time_keyword] <= end)].copy()

    # 固定 y=0，做成“时间轴”
    #df["_y"] = 0
    # 按 filter 映射到不同高度
    if y_map is None:
        y_map = {'UVW2': 0, 'UVM2':1, 'UVW1':2, 'U':3, 'B':4, 'V':5}

    df["_y"] = df[filter_keyword].map(y_map)

    # 如果表里有不在 filter_height_map 中的 filter，防止报错
    df = df.dropna(subset=["_y"])

    # FILTER：紫 -> 橘（可按需微调）
    if color_map is None:
        color_map = {
            "UVW2": "#5B2A86",
            "UVM2": "#3B4CC0",
            "UVW1": "#2C7FB8",
            "U":    "cyan",
            "B":    "#41AB5D",
            "V":    "#F28E2B",
        }

    # DATATYPE：image 圆点，event 十字
    if datatype_keyword is not None and datatype_keyword in df.columns:
        if symbol_map is None:
            symbol_map = {"image": "circle", "event": "cross"}
        fig = px.scatter(
            df,
            x="MIDTIME",
            y="_y",
            color=filter_keyword,
            symbol="DATATYPE",
            color_discrete_map=color_map,
            symbol_map=symbol_map,
            hover_name=obs_col,           # 悬停直接显示 obsid（保留前导0）
            hover_data={"_y": False},     # 不显示 y
        )
    else:
        fig = px.scatter(
            df,
            x="MIDTIME",
            y="_y",
            color=filter_keyword,
            color_discrete_map=color_map,
            hover_name=obs_col,           # 悬停直接显示 obsid（保留前导0）
            hover_data={"_y": False},     # 不显示 y
        )

    fig.update_traces(marker={"size": 9})
    #fig.update_yaxes(visible=False, showticklabels=False, zeroline=False)
    # y 轴显示为 FILTER 分层
    ymin = min(y_map.values()) - 0.5
    ymax = max(y_map.values()) + 0.5
    ymean = (ymin + ymax) / 2
    ystep = (ymax - ymin)
    ymin = ymean - ystep*5
    ymax = ymean + ystep*5
    fig.update_yaxes(
        visible=False,            # 不显示轴，但 range 仍然生效
        range=[ymin, ymax],
        zeroline=False
    )
    fig.update_layout(
        xaxis_title="MIDTIME",
        yaxis_title="",
        title="Observation Timeline",
        legend_title_text="FILTER / DATATYPE",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    if html_out is not None:
        fig.write_html(html_out, include_plotlyjs="cdn")
    return fig