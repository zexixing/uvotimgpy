import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

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
                    arrowprops=dict(facecolor=c, edgecolor=c, width=1, headwidth=6, headlength=5),
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
    # 将 position 从 Axes 坐标 → Data 坐标
    x0_ax, y0_ax = position
    x0_data, y0_data = ax.transAxes.transform((x0_ax, y0_ax))
    
    # 横线像素起止（屏幕坐标）
    half_len = length / 2
    x_start_pix = x0_data - half_len
    x_end_pix = x0_data + half_len
    y_pix = y0_data

    # 上下标签位置（屏幕坐标）
    y_top = y_pix + text_offset
    y_bottom = y_pix - text_offset

    # 反变换为数据坐标
    inv = ax.transData.inverted()
    x_start_data, y_data = inv.transform((x_start_pix, y_pix))
    x_end_data, _ = inv.transform((x_end_pix, y_pix))
    _, y_top_data = inv.transform((x0_data, y_top))
    _, y_bottom_data = inv.transform((x0_data, y_bottom))

    # 画横线
    ax.plot([x_start_data, x_end_data], [y_data, y_data], color=color, linewidth=linewidth)

    # 上标签
    if label_top:
        ax.text((x_start_data + x_end_data)/2, y_top_data, label_top,
                ha='center', va='bottom', color=color, fontsize=fontsize)

    # 下标签
    if label_bottom:
        ax.text((x_start_data + x_end_data)/2, y_bottom_data, label_bottom,
                ha='center', va='top', color=color, fontsize=fontsize)

    return ax


