from typing import Tuple, Union, Optional, List, Callable, Any
import numpy as np
from photutils.aperture import ApertureMask, BoundingBox, Aperture
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from regions import PixelRegion, PixCoord, CirclePixelRegion, RectanglePixelRegion, CircleAnnulusPixelRegion
from functools import reduce
from operator import or_, and_

class RegionConverter:
    @staticmethod
    def region_to_bool_array(region: PixelRegion, 
                             image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Region转换为布尔数组

        Parameters
        ----------
        region : PixelRegion
            regions包的Region对象

        Returns
        -------
        numpy.ndarray
            布尔数组形式的掩膜
        """
        mask_inner = region.to_mask(mode='center').to_image(image_shape)
        return mask_inner.astype(bool)

    @staticmethod
    def aperture_to_bool_array(aperture_mask: ApertureMask, 
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """
        ApertureMask转换为布尔数组

        Parameters
        ----------
        aperture_mask : ApertureMask
            photutils的ApertureMask对象
        image_shape : tuple
            目标图像形状

        Returns
        -------
        numpy.ndarray
            布尔数组形式的掩膜
        """
        full_mask = np.zeros(image_shape, dtype=bool)
        bbox = aperture_mask.bbox
        yslice = slice(bbox.iymin, bbox.iymax)
        xslice = slice(bbox.ixmin, bbox.ixmax)
        full_mask[yslice, xslice] = aperture_mask.data > 0
        return full_mask

    @staticmethod
    def bool_array_to_aperture(bool_array: np.ndarray) -> ApertureMask:
        """
        布尔数组转换为ApertureMask

        Parameters
        ----------
        bool_array : numpy.ndarray
            布尔数组掩膜

        Returns
        -------
        ApertureMask
            photutils的ApertureMask对象
        """
        rows, cols = np.where(bool_array)
        if len(rows) == 0:  # 空掩膜
            return ApertureMask(np.array([[False]]), bbox=BoundingBox(0, 1, 0, 1))
        
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1
        
        mask_data = bool_array[ymin:ymax, xmin:xmax]
        bbox = BoundingBox(ixmin=xmin, ixmax=xmax, iymin=ymin, iymax=ymax)
        
        return ApertureMask(mask_data, bbox=bbox)

    @staticmethod
    def region_to_aperture(region: PixelRegion,
                           image_shape: Tuple[int, int]) -> ApertureMask:
        """
        Region转换为ApertureMask

        Parameters
        ----------
        region : PixelRegion
            regions包的Region对象
        image_shape : tuple
            图像形状

        Returns
        -------
        ApertureMask
            photutils的ApertureMask对象
        """
        bool_array = RegionConverter.region_to_bool_array(region, image_shape)
        return RegionConverter.bool_array_to_aperture(bool_array)
    
    @staticmethod
    def to_bool_array(region: Union[np.ndarray, ApertureMask, PixelRegion], 
                      image_shape: Tuple[int, int]) -> np.ndarray:
        if isinstance(region, ApertureMask):
            return RegionConverter.aperture_to_bool_array(region, image_shape)
        elif isinstance(region, PixelRegion):
            return RegionConverter.region_to_bool_array(region, image_shape)
        else:
            return region
        
    @staticmethod
    def to_bool_array_general(regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                              combine_regions: bool, shape: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """处理输入的regions，转换为布尔数组列表"""
        if isinstance(regions, np.ndarray):
            if regions.dtype != bool:
                raise ValueError("Input array must be boolean type")
            return [regions]
            
        elif isinstance(regions, PixelRegion):
            mask = RegionConverter.region_to_bool_array(regions, shape)
            return [mask]
            
        elif isinstance(regions, (list, tuple)):
            if combine_regions:
                combined_mask = RegionCombiner.union(regions)
                if not isinstance(combined_mask, np.ndarray):
                    combined_mask = RegionConverter.region_to_bool_array(combined_mask, shape)
                return [combined_mask]
            else:
                return [RegionConverter.region_to_bool_array(reg, shape) 
                        for reg in regions]
        else:
            raise ValueError("Unsupported region type")

class RegionCombiner:
    """处理掩膜列表的合并操作"""
    
    @staticmethod
    def _check_masks_type(masks: List[Union[np.ndarray, PixelRegion]]) -> str:
        """
        检查掩膜列表中的数据类型是否一致
        
        Parameters
        ----------
        masks : List[Union[np.ndarray, PixelRegion]]
            掩膜列表
            
        Returns
        -------
        str
            'array' 或 'region'
        """
        if not masks:
            raise TypeError("掩膜列表不能为空")
            
        first_type = type(masks[0])
        if not all(isinstance(mask, first_type) for mask in masks):
            raise TypeError("掩膜列表中的元素类型必须一致")
            
        if isinstance(masks[0], np.ndarray):
            if not all(mask.dtype == bool for mask in masks):
                raise TypeError("NumPy数组掩膜必须是布尔类型")
            return 'array'
        elif isinstance(masks[0], PixelRegion):
            return 'region'
        else:
            raise TypeError("不支持的掩膜类型")
    
    @staticmethod
    def _check_array_shapes(masks: List[np.ndarray]) -> None:
        """检查数组掩膜的形状是否一致"""
        if not all(mask.shape == masks[0].shape for mask in masks):
            raise ValueError("所有数组掩膜的形状必须相同")
    
    @classmethod
    def union(cls, 
             masks: List[Union[np.ndarray, PixelRegion]]) -> Union[np.ndarray, PixelRegion]:
        """
        计算掩膜列表的并集
        
        Parameters
        ----------
        masks : List[Union[np.ndarray, PixelRegion]]
            要合并的掩膜列表
            
        Returns
        -------
        Union[np.ndarray, PixelRegion]
            合并后的掩膜
        """
        mask_type = cls._check_masks_type(masks)
        
        if mask_type == 'array':
            cls._check_array_shapes(masks)
            return reduce(or_, masks)
        else:  # region
            return reduce(or_, masks)
    
    @classmethod
    def intersection(cls, 
                    masks: List[Union[np.ndarray, PixelRegion]]) -> Union[np.ndarray, PixelRegion]:
        """
        计算掩膜列表的交集
        
        Parameters
        ----------
        masks : List[Union[np.ndarray, PixelRegion]]
            要合并的掩膜列表
            
        Returns
        -------
        Union[np.ndarray, PixelRegion]
            合并后的掩膜
        """
        mask_type = cls._check_masks_type(masks)
        
        if mask_type == 'array':
            cls._check_array_shapes(masks)
            return reduce(and_, masks)
        else:  # region
            return reduce(and_, masks)

def mask_image(image: np.ndarray,
               bad_pixel_mask: Optional[Union[np.ndarray, ApertureMask, PixelRegion]]) -> np.ndarray:
    """
    处理输入图像和掩模
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    bad_pixel_mask : np.ndarray or ApertureMask or PixelRegion, optional
        坏像素掩模，True表示被mask的像素
        
    Returns
    -------
    np.ndarray
        处理后的图像，被mask的像素设为nan
    """
    if bad_pixel_mask is not None:
        mask = RegionConverter.to_bool_array(bad_pixel_mask, image.shape)
        masked_image = image.copy()
        masked_image[mask] = np.nan
        return masked_image
    return image

class RegionSelector:
    def __init__(self, image_data, vmin=0, vmax=None, 
                 row_range=None, col_range=None, shape='circle', region_plot=None):
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
        self.current_size = 5
        self.shape = shape
        
        # Display parameters
        self.vmin = vmin
        self.vmax = vmax if vmax is not None else np.percentile(self.image, 99)
        
        # Display image and cursor
        self.display = self.ax.imshow(self.image, origin='lower', cmap='viridis',
                                    vmin=self.vmin, vmax=self.vmax,
                                    extent=[-0.5, self.image.shape[1]-0.5, -0.5, self.image.shape[0]-0.5])
        self.colorbar = plt.colorbar(self.display)
        if region_plot is not None:
            if isinstance(region_plot, list):
                for region_item in region_plot:
                    region_item.plot(ax=self.ax, color='orange', linestyle='--', lw=1, alpha=0.7)
            else:
                region_item = region_plot
                region_item.plot(ax=self.ax, color='orange', linestyle='--', lw=1, alpha=0.7)
        
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
            'Left Click: Select Region  A: Toggle Circle/Square\n'
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
            
    def get_regions(self):
        plt.show()
        return self.regions

def save_regions(regions: List[PixelRegion], file_path: str, correct: Optional[float] = 1) -> None:
    """
    1: 将从图上选择的PixelRegion列表保存为DS9可读取的.reg文件
    """
    with open(file_path, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" ")
        f.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        f.write("physical\n")
        
        for region in regions:
            if isinstance(region, CirclePixelRegion):
                # Python (col, row) -> DS9 (x, y)
                x_ds9 = region.center.x + correct  # col -> x
                y_ds9 = region.center.y + correct  # row -> y
                r = region.radius
                f.write(f"circle({x_ds9},{y_ds9},{r})\n")
            elif isinstance(region, RectanglePixelRegion):
                x_ds9 = region.center.x + correct  # col -> x
                y_ds9 = region.center.y + correct  # row -> y
                w, h = region.width, region.height
                f.write(f"box({x_ds9},{y_ds9},{w},{h},0)\n")

def load_regions(file_path: str, shape: Tuple[int, int] = None, correct: Optional[float] = -1) -> Union[List[PixelRegion], np.ndarray]:
    """
    -1: 读取DS9的.reg文件，返回PixelRegion列表或布尔数组到array处理
    """
    regions = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not any('physical' in line.lower() for line in lines):
        raise ValueError("只支持physical坐标系统的region文件")
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('global') or not line or line == 'physical':
            continue
            
        if line.startswith('circle'):
            params = line[line.find('(')+1:line.find(')')].split(',')
            # DS9 (x, y) -> Python (col, row)
            col = float(params[0]) + correct  # x -> col
            row = float(params[1]) + correct  # y -> row
            r = float(params[2])
            center = PixCoord(col, row)
            regions.append(CirclePixelRegion(center, r))
            
        elif line.startswith('box'):
            params = line[line.find('(')+1:line.find(')')].split(',')
            col = float(params[0]) + correct  # x -> col
            row = float(params[1]) + correct  # y -> row
            w = float(params[2])
            h = float(params[3])
            center = PixCoord(col, row)
            regions.append(RectanglePixelRegion(center, w, h))
    
    if shape is not None:
        combined_regions = RegionCombiner.union(regions)
        mask = RegionConverter.region_to_bool_array(combined_regions, image_shape=shape)
        return mask
    else:
        return regions

def adjust_regions(regions_list, old_coord, new_coord):
    # 计算偏移量
    offset_col = new_coord[0] - old_coord[0]
    offset_row = new_coord[1] - old_coord[1]
    # 创建新的regions列表
    new_regions = []
    for reg in regions_list:
        # 深拷贝region以避免修改原始数据
        new_reg = reg.copy()
        # 调整位置
        new_col = new_reg.center.x + offset_col
        new_row = new_reg.center.y + offset_row
        new_reg.center = PixCoord(x=new_col, y=new_row)
        new_regions.append(new_reg)
    return new_regions

def get_total_bounds(region_list):
    # 获取每个圆形region的边界框
    bounds = [region.bounding_box for region in region_list]
    
    # 从边界框中提取最小最大值
    col_mins = [box.ixmin for box in bounds]
    col_maxs = [box.ixmax for box in bounds]
    row_mins = [box.iymin for box in bounds]
    row_maxs = [box.iymax for box in bounds]
    
    total_row_min = min(row_mins)
    total_row_max = max(row_maxs)
    total_col_min = min(col_mins)
    total_col_max = max(col_maxs)
    
    return total_row_min, total_row_max, total_col_min, total_col_max

class RegionStatistics:
    """区域统计量计算类"""
    
    @staticmethod
    def calculate_stats(data: np.ndarray,
                       regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                       func: Callable[[np.ndarray], Any],
                       combine_regions: bool = False,
                       mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
                       ) -> Union[Any, List[Any]]:
        """
        计算区域统计量的静态方法
        
        Parameters
        ----------
        data : np.ndarray
            要统计的数据数组
        regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
            要统计的区域，可以是单个PixelRegion，PixelRegion列表，或布尔数组
        stat_func : Callable[[np.ndarray], Any]
            用于计算统计量的函数，接收有效数据数组作为输入
        combine_regions : bool, optional
            当输入多个区域时，是否将它们合并统计，默认False（分别统计）
        mask : Union[PixelRegion, List[PixelRegion], np.ndarray], optional
            需要排除的区域，可以是区域对象、区域列表或布尔掩模
        """
        bool_masks = RegionConverter.to_bool_array_general(regions, combine_regions=combine_regions, shape=data.shape)
        
        # 处理mask
        exclude_mask = None
        if mask is not None:
            exclude_masks = RegionConverter.to_bool_array_general(mask, combine_regions=True, shape=data.shape)
            exclude_mask = exclude_masks[0]
            
        stats = [RegionStatistics._calculate_stat(data, region_mask, func, exclude_mask) 
                for region_mask in bool_masks]
        return stats[0] if len(stats) == 1 else stats

    @staticmethod
    def _calculate_stat(data: np.ndarray, 
                       mask: np.ndarray, 
                       func: Callable[[np.ndarray], Any],
                       exclude_mask: Optional[np.ndarray] = None) -> Any:
        """计算单个区域的统计量"""
        if exclude_mask is not None:
            valid_mask = mask & ~exclude_mask & ~np.isnan(data)
        else:
            valid_mask = mask & ~np.isnan(data)
        valid_data = data[valid_mask]
        return func(valid_data)

    # 快捷方法
    @staticmethod
    def count_pixels(data: np.ndarray, 
                    regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                    combine_regions: bool = False,
                    mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
                    ) -> Union[int, List[int]]:
        """计算区域内的像素数量"""
        return RegionStatistics.calculate_stats(data, regions, len, combine_regions, mask)

    @staticmethod
    def sum(data: np.ndarray, 
           regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
           combine_regions: bool = False,
           mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
           ) -> Union[float, List[float]]:
        """计算区域内像素值的和"""
        return RegionStatistics.calculate_stats(data, regions, np.sum, combine_regions, mask)

    @staticmethod
    def sum_square(data: np.ndarray, 
                   regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                   combine_regions: bool = False,
                   mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
                   ) -> Union[float, List[float]]:
        """计算区域内像素值平方的和"""
        return RegionStatistics.calculate_stats(data*data, regions, np.sum, combine_regions, mask)

    @staticmethod
    def mean(data: np.ndarray, 
            regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
            combine_regions: bool = False,
            mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
            ) -> Union[float, List[float]]:
        """计算区域内像素值的平均值"""
        return RegionStatistics.calculate_stats(data, regions, np.mean, combine_regions, mask)

    @staticmethod
    def median(data: np.ndarray, 
              regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
              combine_regions: bool = False,
              mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
              ) -> Union[float, List[float]]:
        """计算区域内像素值的中位数"""
        return RegionStatistics.calculate_stats(data, regions, np.median, combine_regions, mask)

    @staticmethod
    def std(data: np.ndarray, 
           regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
           combine_regions: bool = False,
           mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
           ) -> Union[float, List[float]]:
        """计算区域内像素值的标准差"""
        return RegionStatistics.calculate_stats(data, regions, np.std, combine_regions, mask)

    @staticmethod
    def min(data: np.ndarray, 
           regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
           combine_regions: bool = False,
           mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
           ) -> Union[float, List[float]]:
        """计算区域内像素值的最小值"""
        return RegionStatistics.calculate_stats(data, regions, np.min, combine_regions, mask)

    @staticmethod
    def max(data: np.ndarray, 
           regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
           combine_regions: bool = False,
           mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None
           ) -> Union[float, List[float]]:
        """计算区域内像素值的最大值"""
        return RegionStatistics.calculate_stats(data, regions, np.max, combine_regions, mask)
    
def create_circle_region(center, radius):
    center = PixCoord(x=center[0], y=center[1])
    circle_region = CirclePixelRegion(center=center, radius=radius)
    return circle_region

def create_crcle_annulus_region(center, inner_radius, outer_radius):
    # center in (col, row)
    center = PixCoord(x=center[0], y=center[1])
    annulus_region = CircleAnnulusPixelRegion(center=center, 
                                             inner_radius=inner_radius,
                                             outer_radius=outer_radius)
    return annulus_region