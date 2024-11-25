from typing import Tuple, Union, Optional
import numpy as np
from photutils.aperture import ApertureMask, BoundingBox
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from regions import PixCoord, CirclePixelRegion, RectanglePixelRegion


class UnifiedMask:
    def __init__(self, mask_data: Union[np.ndarray, ApertureMask], image_shape: Tuple[int, int] = None):
        """
        统一的掩膜类
        
        Parameters
        ----------
        mask_data : numpy.ndarray 或 ApertureMask
            掩膜数据
        image_shape : tuple, optional
            原始图像的形状，当使用ApertureMask时必须提供
        """
        if isinstance(mask_data, ApertureMask):
            if image_shape is None:
                raise ValueError("image_shape must be provided when using ApertureMask")
            self._mask = mask_data
            self._image_shape = image_shape
            self._is_aperture = True
        else:
            self._mask = np.asarray(mask_data, dtype=bool)
            self._image_shape = mask_data.shape
            self._is_aperture = False
    
    def to_bool_array(self) -> np.ndarray:
        """
        转换为布尔数组
        
        Returns
        -------
        numpy.ndarray
            布尔数组形式的掩膜
        """
        if not self._is_aperture:
            return self._mask
        
        full_mask = np.zeros(self._image_shape, dtype=bool)
        bbox = self._mask.bbox
        yslice = slice(bbox.iymin, bbox.iymax)
        xslice = slice(bbox.ixmin, bbox.ixmax)
        full_mask[yslice, xslice] = self._mask.data > 0
        return full_mask
    
    def to_aperture_mask(self) -> ApertureMask:
        """
        转换为ApertureMask
        
        Returns
        -------
        ApertureMask
            photutils的ApertureMask对象
        """
        if self._is_aperture:
            return self._mask
        
        # 找到掩膜的边界框
        rows, cols = np.where(self._mask)
        if len(rows) == 0:  # 空掩膜
            return ApertureMask(np.array([[False]]), bbox=BoundingBox(0, 1, 0, 1))
        
        # 计算边界框
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1
        
        # 提取边界框内的数据
        mask_data = self._mask[ymin:ymax, xmin:xmax]
        bbox = BoundingBox(ixmin=xmin, ixmax=xmax, iymin=ymin, iymax=ymax)
        
        return ApertureMask(mask_data, bbox=bbox)
    
    def __array__(self) -> np.ndarray:
        """使对象可以直接用作numpy数组"""
        return self.to_bool_array()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """返回掩膜形状"""
        return self._image_shape
    
def mask_image(image: np.ndarray,
               bad_pixel_mask: Optional[Union[np.ndarray, ApertureMask]]) -> np.ndarray:
    """
    处理输入图像和掩模
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    bad_pixel_mask : np.ndarray or ApertureMask, optional
        坏像素掩模，True表示被mask的像素
        
    Returns
    -------
    np.ndarray
        处理后的图像，被mask的像素设为nan
    """
    if bad_pixel_mask is not None:
        mask = UnifiedMask(bad_pixel_mask, image.shape)
        masked_image = image.copy()
        masked_image[mask.to_bool_array()] = np.nan
        return masked_image
    return image

class ApertureSelector:
    def __init__(self, image_data, vmin=0, vmax=None, 
                 row_range=None, col_range=None, shape='circle'):
        """
        Parameters
        ----------
        image_data : numpy.ndarray
            输入图像数据
        vmin, vmax : float, optional
            显示范围
        row_range : tuple, optional
            显示的行范围，格式为(start, end)
        col_range : tuple, optional
            显示的列范围，格式为(start, end)
        shape : str, optional
            选择区域的形状，'circle' 或 'square'
        """
        self.image = image_data
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_adjustable('box')
        self.ax.set_aspect('equal')
    
        self.regions = []
        self.patches = []
        self.current_size = 10
        self.shape = shape
        
        # Display parameters
        self.vmin = vmin
        self.vmax = vmax if vmax is not None else np.percentile(self.image, 99)
        
        # Display image and cursor
        self.display = self.ax.imshow(self.image, origin='lower', cmap='viridis',
                                    vmin=self.vmin, vmax=self.vmax)
        self.colorbar = plt.colorbar(self.display)
        
        # Set display range if provided
        if row_range is not None:
            self.ax.set_ylim(row_range)
        if col_range is not None:
            self.ax.set_xlim(col_range)
        
        # Preview patch (initially invisible)
        center = (self.image.shape[1]/2, self.image.shape[0]/2)  # (col, row)
        if self.shape == 'circle':
            self.preview_patch = Circle(center, self.current_size, 
                                      fill=False, color='red', linestyle='--', 
                                      alpha=0, visible=False)
        else:
            size = self.current_size * 2
            self.preview_patch = Rectangle((center[0]-self.current_size, center[1]-self.current_size),
                                        size, size, fill=False, color='red', linestyle='--',
                                        alpha=0, visible=False)
        self.ax.add_patch(self.preview_patch)
        
        # Set title with instructions
        self.instruction_text = (
            'Left Click: Select Aperture  A: Toggle Circle/Square\n'
            'W/E: Decrease/Increase Size  V/B: Decrease/Increase Min  N/M: Decrease/Increase Max\n'
            'Z: Undo  R: Reset View  Enter: Finish\n'
            'Arrow Keys: Pan View  I/O: Zoom In/Out'
        )
        self.status_text = f'Current size: {self.current_size:.1f} | Shape: {self.shape}'
        self._update_title()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self._onkey)
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)

    def _update_title(self):
        """更新标题，包括说明和状态信息"""
        full_title = f'{self.instruction_text}\n{self.status_text}'
        self.ax.set_title(full_title)
        
    def _update_display(self):
        self.display.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.fig.canvas.draw()
        
    def _show_preview(self):
        # Get current view center
        col_lim = self.ax.get_xlim()
        row_lim = self.ax.get_ylim()
        center_col = (col_lim[1] + col_lim[0]) / 2
        center_row = (row_lim[1] + row_lim[0]) / 2
        
        # Remove old preview patch
        self.preview_patch.remove()
        
        # Create new preview patch
        if self.shape == 'circle':
            self.preview_patch = Circle((center_col, center_row), self.current_size,
                                     fill=False, color='red', linestyle='--', alpha=0.8)
        else:
            size = self.current_size * 2
            self.preview_patch = Rectangle((center_col-self.current_size, center_row-self.current_size),
                                        size, size, fill=False, color='red', linestyle='--', alpha=0.8)
        
        self.ax.add_patch(self.preview_patch)
        self.fig.canvas.draw()
        
        # Flash effect
        try:
            plt.pause(0.1)
        except:
            pass
        self.preview_patch.set_alpha(0)
        self.fig.canvas.draw()
        
    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            col, row = event.xdata, event.ydata
            center = PixCoord(x=col, y=row)
            
            if self.shape == 'circle':
                region = CirclePixelRegion(center=center, radius=self.current_size)
                patch = Circle((col, row), self.current_size, 
                             fill=False, color='red', alpha=0.5)
            else:
                # 创建方形区域
                region = RectanglePixelRegion(
                    center=center,
                    width=self.current_size * 2,
                    height=self.current_size * 2
                )
                patch = Rectangle((col-self.current_size, row-self.current_size),
                                self.current_size * 2, self.current_size * 2,
                                fill=False, color='red', alpha=0.5)
            
            self.regions.append(region)
            self.patches.append(patch)
            self.ax.add_patch(patch)
            self.fig.canvas.draw()
            
    def _onkey(self, event):
        if event.key == 'z':  # Undo
            if self.patches:
                self.patches[-1].remove()
                self.patches.pop()
                self.regions.pop()
                self.fig.canvas.draw()
                
        elif event.key == 'a':  # Toggle shape (changed from 's' to 'a')
            self.shape = 'square' if self.shape == 'circle' else 'circle'
            self.status_text = f'Current size: {self.current_size:.1f} | Shape: {self.shape}'
            self._update_title()
            self._show_preview()  # 添加形状切换的视觉提示
                
        elif event.key == 'enter':  # Finish
            plt.close()
            
        elif event.key == 'r':  # Reset view
            self.ax.set_xlim(0, self.image.shape[1])
            self.ax.set_ylim(0, self.image.shape[0])
            self.fig.canvas.draw()
            
        elif event.key in ['left', 'right', 'up', 'down']:  # Pan view
            curr_col_lim = self.ax.get_xlim()
            curr_row_lim = self.ax.get_ylim()
            
            col_width = abs(curr_col_lim[1] - curr_col_lim[0])
            row_height = abs(curr_row_lim[1] - curr_row_lim[0])

            col_move = col_width * 0.1
            row_move = row_height * 0.1
            
            if event.key == 'left':
                new_col_lim = (curr_col_lim[0] + col_move, curr_col_lim[1] + col_move)
                self.ax.set_xlim(new_col_lim)
            elif event.key == 'right':
                new_col_lim = (curr_col_lim[0] - col_move, curr_col_lim[1] - col_move)
                self.ax.set_xlim(new_col_lim)
            elif event.key == 'up':
                new_row_lim = (curr_row_lim[0] - row_move, curr_row_lim[1] - row_move)
                self.ax.set_ylim(new_row_lim)
            elif event.key == 'down':
                new_row_lim = (curr_row_lim[0] + row_move, curr_row_lim[1] + row_move)
                self.ax.set_ylim(new_row_lim)

            self.fig.canvas.draw()
            
        elif event.key in ['i', 'o']:  # Zoom
            col_lim = self.ax.get_xlim()
            row_lim = self.ax.get_ylim()
            center_col = (col_lim[1] + col_lim[0]) / 2
            center_row = (row_lim[1] + row_lim[0]) / 2
            width = col_lim[1] - col_lim[0]
            height = row_lim[1] - row_lim[0]
            
            if event.key == 'i':  # Zoom in
                factor = 0.8
            else:  # Zoom out
                factor = 1.25
                
            self.ax.set_xlim(center_col - width/2 * factor, 
                            center_col + width/2 * factor)
            self.ax.set_ylim(center_row - height/2 * factor, 
                            center_row + height/2 * factor)
            self.fig.canvas.draw()
            
        elif event.key in ['v', 'b', 'n', 'm']:  # Adjust display range
            if event.key == 'v':
                self.vmin -= self.vmax * 0.05
            elif event.key == 'b':
                self.vmin = min(self.vmax, self.vmin + self.vmax * 0.05)
            elif event.key == 'n':
                self.vmax = max(self.vmin, self.vmax - self.vmax * 0.05)
            elif event.key == 'm':
                self.vmax += self.vmax * 0.05
            self._update_display()
            
        elif event.key in ['w', 'e']:  # Adjust size
            if event.key == 'w':
                self.current_size = max(1, self.current_size - 0.5)
            elif event.key == 'e':
                self.current_size += 0.5
            #print(f"Current size: {self.current_size:.1f}")
            self.status_text = f'Current size: {self.current_size:.1f} | Shape: {self.shape}'
            self._update_title()
            self._show_preview()
            
    def get_apertures(self):
        plt.show()
        return self.regions

if __name__ == '__main__':
    image = np.random.normal(0, 1, (100, 100))
    bool_mask = image > 0.5
    mask1 = UnifiedMask(bool_mask)
    aperture_mask1 = mask1.to_aperture_mask()  # 转换为ApertureMask