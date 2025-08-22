from typing import Union, Tuple, List, Optional, Dict, Any
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import matplotlib.pyplot as plt
from regions import CirclePixelRegion, PixCoord, PixelRegion
from photutils.aperture import ApertureMask
from uvotimgpy.utils.image_operation import calc_radial_profile, DistanceMap, ImageDistanceCalculator
from uvotimgpy.query import StarCatalogQuery
from uvotimgpy.base.region import RegionConverter, RegionCombiner, RegionSelector, save_regions, select_mask_regions, expand_shrink_region, get_exclude_region
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from skimage import restoration
from uvotimgpy.base.visualizer import MaskInspector

class StarIdentifier:
    """Identify stars, cosmic rays and other pixels that need to be removed from the image"""
    
    def __init__(self):
        self.last_mask = None  # Store the most recently identified mask
    
    def by_comparison(self, image1: np.ndarray, image2: np.ndarray,
                     threshold: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
        """Identify by comparing two images"""
        diff = image1 - image2
        mask_pos = diff > threshold
        mask_neg = diff < -threshold
        self.last_mask = mask_pos | mask_neg
        return mask_pos, mask_neg # mask represents stars
    
    #def by_sigma_clip(self, image: np.ndarray, sigma: float = 3.,
    #                 maxiters: Optional[int] = 3,
    #                 exclude_region: Optional[Union[np.ndarray, ApertureMask, PixelRegion]] = None) -> np.ndarray:
    #    """Identify using sigma-clip method"""
    #    mask = np.zeros_like(image, dtype=bool)
    #    if exclude_region is not None:
    #        region_mask = RegionConverter.to_bool_array(exclude_region, image.shape)
    #        valid_pixels = ~region_mask
    #        clipped = sigma_clip(image[valid_pixels], sigma=sigma, maxiters=maxiters, masked=True)
    #        mask[valid_pixels] = clipped.mask
    #    else:
    #        clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters, masked=True)
    #        mask = clipped.mask
    #    self.last_mask = mask
    #    return mask # mask represents stars
    
    def by_sigma_clip(self, image: np.ndarray, sigma: float = 3.,
                      maxiters: Optional[int] = 3,
                      exclude_region: Optional[Union[np.ndarray, ApertureMask, PixelRegion]] = None,
                      tile_size: Optional[int] = None,
                      area_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
                      expand_shrink_paras: Optional[Dict[str, Any]] = None
                      ) -> np.ndarray:
        """
        Identify cosmic rays using sigma-clip, optionally in tiles.
        area_size: (min_area, max_area)
        expand_shrink_paras: {'radius': int, 'method': 'expand' or 'shrink', 'speed': 'normal' or 'fast'}
        """
        mask = np.zeros_like(image, dtype=bool)
        ny, nx = image.shape

        # 如果需要 exclude 区域
        if exclude_region is not None:
            region_mask = RegionConverter.to_bool_array(exclude_region, image.shape)
        else:
            region_mask = np.zeros_like(image, dtype=bool)

        nan_mask = np.isnan(image)

        if tile_size is None:
            # 全图 clip
            valid_pixels = (~region_mask) & (~nan_mask)
            clipped = sigma_clip(image[valid_pixels], sigma=sigma, maxiters=maxiters, masked=True)
            mask[valid_pixels] = clipped.mask
        else:
            # 分块 clip
            for y0 in range(0, ny, tile_size):
                for x0 in range(0, nx, tile_size):
                    y1 = min(y0 + tile_size, ny)
                    x1 = min(x0 + tile_size, nx)
                    tile = image[y0:y1, x0:x1]
                    tile_mask = region_mask[y0:y1, x0:x1]
                    tile_nan = np.isnan(tile)
                    valid_pixels = (~tile_mask) & (~tile_nan)

                    if np.any(valid_pixels):
                        clipped = sigma_clip(tile[valid_pixels], sigma=sigma, maxiters=maxiters, masked=True)
                        submask = np.zeros_like(tile, dtype=bool)
                        submask[valid_pixels] = clipped.mask
                        mask[y0:y1, x0:x1] = submask
                    # 否则该 tile 无有效 pixel，跳过
        if area_size is not None:
            min_area, max_area = area_size
            mask = select_mask_regions(mask, min_area=min_area, max_area=max_area)
        if expand_shrink_paras is not None:
            radius = expand_shrink_paras.get('radius', 2)
            method = expand_shrink_paras.get('method', 'expand')
            speed = expand_shrink_paras.get('speed', 'normal')
            mask = expand_shrink_region(mask, radius=radius, method=method, speed=speed)
        self.last_mask = mask
        return mask  # True 表示检测为异常值，NaN 像素恒为 False
    
    def by_manual(self, image: np.ndarray, 
                  row_range: Optional[Tuple[int, int]] = None,
                  col_range: Optional[Tuple[int, int]] = None,
                  vmin = 0, vmax=2,
                  save_path: Optional[Union[str, Path]] = None,
                  region_plot: Optional[Union[PixelRegion, List[PixelRegion]]] = None,
                  ) -> np.ndarray:
        """Identify by manual input"""
        print("Creating selector...")
        
        # Ensure all previous windows are closed
        plt.close('all')
        
        # Create selector
        selector = RegionSelector(image, vmin=vmin, vmax=vmax, 
                                row_range=row_range, 
                                col_range=col_range,
                                region_plot=region_plot)
        
        print("Getting regions...")
        # Explicitly call show and wait for window to close
        plt.show(block=True)
        
        # Get regions
        regions = selector.get_regions()
        print("Regions obtained.")#, regions)
        
        if not regions:  # If no regions were selected
            return np.zeros_like(image, dtype=bool)
        
        # Combine all selected regions
        if save_path is not None:
            save_regions(regions=regions, file_path=save_path, correct=1)
        
        # Convert to boolean array
        combined_regions = RegionCombiner.union(regions)
        mask = RegionConverter.region_to_bool_array(combined_regions, image_shape=image.shape)
        
        self.last_mask = mask
        return mask
    
    def by_catalog(self, image: np.ndarray, wcs: WCS, mag_limit: float = 15,
                  catalog: str = 'GSC', aperture_radius: float = 5) -> np.ndarray:
        """Create star mask"""
        # catalog: GSC, GAIA, UCAC4, APASS, USNOB, SIMBAD

        # Calculate celestial coordinates for the four corners of the image
        n_rows, n_cols = image.shape
        center = np.array([n_cols/2, n_rows/2])  # (col, row)
        center_sky = wcs.pixel_to_world(center[0], center[1])

        # Calculate maximum distance from center to edge (in pixels, assuming row and col directions)
        max_dist = ImageDistanceCalculator.from_edges(image, center, distance_method='max', wcs=wcs)

        radius = 1.1 * max_dist

        # Query star catalog
        catalog_query = StarCatalogQuery(center_sky, radius, mag_limit)
        stars, ra_key, dec_key = catalog_query.query(catalog)

        # Create SkyCoord object, transform coordinates and create mask
        coords = SkyCoord(ra=stars[ra_key], dec=stars[dec_key])
        pixel_coords = wcs.world_to_pixel(coords)
        positions = np.array(pixel_coords).T

        # Filter stars within image boundaries
        valid_stars = (
            (positions[:, 0] >= 0) & 
            (positions[:, 0] < image.shape[1]) & 
            (positions[:, 1] >= 0) & 
            (positions[:, 1] < image.shape[0])
        )
        positions = positions[valid_stars]

        # Create circular aperture
        #apertures = CircularAperture(positions, r=aperture_radius)
        # Create mask
        #mask = np.zeros(image.shape, dtype=bool)
        #masks = apertures.to_mask(method='center')

        #for mask_obj in masks:
        #    # Get mask position information
        #    slices = mask_obj.get_overlap_slices(image.shape)
        #    if slices is not None:
        #        data_slc, mask_slc = slices
        #        mask[data_slc] |= mask_obj.data[mask_slc] > 0
        mask = np.zeros(image.shape, dtype=bool)

        centers = PixCoord(positions[:, 0], positions[:, 1])
        circles = [CirclePixelRegion(center=center, radius=aperture_radius) for center in centers]
        combined_regions = RegionCombiner.union(circles)
        mask = RegionConverter.region_to_bool_array(combined_regions, image_shape=image.shape)

        self.last_mask = mask
        return mask

class PixelFiller:
    """Fill marked pixels"""
    
    def by_comparison(self, image1: np.ndarray, image2: np.ndarray,
                     mask_pos: np.ndarray, mask_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill using mutual comparison of two images"""
        filled1 = image1.copy()
        filled2 = image2.copy()
        
        filled1[mask_pos] = image2[mask_pos]
        filled2[mask_neg] = image1[mask_neg]
        
        return filled1, filled2
    
    def by_rings(self, image: np.ndarray, mask: np.ndarray, 
                 center: tuple, step: float, 
                 method: str = 'median',
                 start: Optional[float] = None,
                 end: Optional[float] = None,) -> np.ndarray:
        """
        Fill masked pixels using ring-shaped regions

        Parameters
        ----------
        image : np.ndarray
            Input image
        mask : np.ndarray
            Bad pixel mask, True represents masked pixels
        center : tuple
            Ring center coordinates (col, row)
        step : float
            Ring step size
        method : str
            Calculation method, 'median' or 'mean'
        start, end : float, optional
            Start and end radii of rings

        Returns
        -------
        np.ndarray
            Filled image
        """
        # Parameter checks
        if mask.shape != np.asarray(image).shape:
            raise ValueError("image and mask must have the same shape")
        if method not in ['median', 'mean']:
            raise ValueError("method must be 'median' or 'mean'")
        if step <= 0:
            raise ValueError("step must be positive")

        # If no masked pixels, return original image
        if not np.any(mask):
            return image.copy()
        
        # Initialize
        filled_image = image.copy()

        # Use calc_radial_profile to calculate values for each ring
        profile = calc_radial_profile(image, center=center, step=step, bad_pixel_mask=mask,
                                      start=start, end=end, method=method)
        radii, values = profile.get_radial_profile()
        dist_map = DistanceMap(image, center)

        # Process each valid ring
        for r, v in zip(radii, values):
            # Create mask for current ring
            ring_mask = dist_map.get_range_mask(r-step/2, r+step/2)

            # Find masked pixels in this ring
            masked_pixels = mask & ring_mask

            # If there are masked pixels in this ring, fill with calculated value
            if np.any(masked_pixels):
                filled_image[masked_pixels] = v

        return filled_image, radii, values
    
    def _iterative_fill(self, data: np.ndarray, mask: np.ndarray, 
                        filter_func, footprint: np.ndarray) -> np.ndarray:
        """Generic iterative filling function
        
        Parameters
        ----------
        data : np.ndarray
            Data to be filled (can be image or error array)
        mask : np.ndarray
            Pixel mask for areas to be filled
        filter_func : callable
            Function used to calculate fill values (e.g., np.nanmedian or error_propagation)
        footprint : np.ndarray
            Structuring element that defines the neighborhood
            
        Returns
        -------
        np.ndarray
            Filled array
        """
        # Initialize array with NaN
        working_data = data.copy()
        working_data[mask] = np.nan
        
        # Iterate filling until no more changes or maximum iterations reached
        max_iters = 10
        for _ in range(max_iters):
            filled_values = ndimage.generic_filter(
                working_data,
                function=filter_func,
                footprint=footprint,
                mode='constant',
                cval=np.nan
            )
            
            # Only update masked regions
            new_data = working_data.copy()
            new_data[mask] = filled_values[mask]
            
            # Check for convergence (if all values are filled)
            if not np.any(np.isnan(new_data[mask])):
                working_data = new_data
                break
                
            # Check if there are any changes
            if np.allclose(new_data, working_data, equal_nan=True):
                break
                
            working_data = new_data
        
        return working_data

    def by_neighbors(self, image: np.ndarray, mask: np.ndarray, 
                    radius: int = 4, method: str = 'nearest',
                    error: Optional[np.ndarray] = None,
                    mean_or_median: str = 'median') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fill masked pixels using neighboring pixels
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        mask : np.ndarray
            Bad pixel mask, True represents masked pixels
        radius : int
            Neighborhood radius (used only when method='nearest'), default is 4
        method : str
            Filling method, options:
            - 'nearest': Nearest neighbor interpolation (default), fills using neighborhood median
            - 'biharmonic': Biharmonic interpolation, suitable for smooth filling
        error : np.ndarray, optional
            Error array for input image. If provided, will calculate error propagation for filled pixels
            
        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            If error parameter is not provided, returns filled image;
            If error parameter is provided, returns (filled_image, filled_error) tuple
        """
        # Parameter checks
        if mask.shape != image.shape:
            raise ValueError("image and mask must have the same shape")
        if method not in ['nearest', 'median_filter','uniform_filter']:
            raise ValueError("method must be one of: 'nearest', 'median_filter', 'uniform_filter'")
        if error is not None and error.shape != image.shape:
            raise ValueError("error array must have the same shape as image")
        
        # If no masked pixels, return original image
        if not np.any(mask):
            if error is not None:
                return image.copy(), error.copy()
            return image.copy()
        
        # Initialize output
        filled_image = image.copy()
        filled_error = error.copy() if error is not None else None
        
        if method == 'nearest':
            footprint = ndimage.generate_binary_structure(2, 1)
            footprint = ndimage.iterate_structure(footprint, radius)
            
            # Use generic filling function to fill the image
            if mean_or_median == 'median':
                fill_func = np.nanmedian
            else:
                fill_func = np.nanmean
            filled_image = self._iterative_fill(
                image, mask, fill_func, footprint
            )
            
            # Calculate error (if needed)
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='nearest', footprint=footprint
                )
        elif method == 'median_filter':
            filtered = ndimage.median_filter(image, size=2 * radius + 1)
            filled_image = image.copy()
            filled_image[mask] = filtered[mask]
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='nearest', footprint=footprint
                )
        elif method == 'uniform_filter':
            #filtered = ndimage.uniform_filter(image, size=2 * radius + 1)
            filtered = image.copy()
            filtered[mask] = np.nan
            filtered = np.nan_to_num(filtered, nan=0.0)
            num = ndimage.uniform_filter(filtered, size=2 * radius + 1, mode='constant', cval=0.0)
            denom = ndimage.uniform_filter(mask, size=2 * radius + 1, mode='constant', cval=0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered = num/denom
            filled_image = image.copy()
            filled_image[mask] = filtered[mask]
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='nearest', footprint=footprint
                )
        elif method == 'convolution':
            image_with_nans = image.copy()
            image_with_nans[mask] = np.nan
            kernel = Gaussian2DKernel(x_stddev=radius)
            filled_image = interpolate_replace_nans(image_with_nans, kernel)
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='nearest', footprint=footprint
                )
        elif 'griddata' in method: # e.g., griddata_nearest
            griddata_method = method.split('_')[1]
            row, col = np.indices(self.image.shape)
            points = np.array((col[~mask], row[~mask])).T
            values = image[~mask]
            filled_image = griddata(points, values, (col, row), method=griddata_method)
            return filled_image
        elif method == 'generic_filter':
            def nan_median_filter(values):
                valid_values = values[~np.isnan(values)]
                return np.median(valid_values) if len(valid_values) > 0 else np.nan
            image_with_nans = self.image.copy()
            image_with_nans[self.mask] = np.nan
            filled_image = generic_filter(image_with_nans, nan_median_filter, size=radius)
            return filled_image
        else:  # 'biharmonic'
            from skimage.restoration import inpaint
            # Biharmonic interpolation, suitable for smooth filling
            filled_image = inpaint.inpaint_biharmonic(
                image, mask
            )
            
            # Calculate error (if needed)
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='biharmonic'
                )
        
        if error is not None:
            return filled_image, filled_error
        return filled_image
    
    def _by_neighbors_calculate_error(self, error: np.ndarray, mask: np.ndarray, 
                                        method: str = 'nearest', **kwargs) -> np.ndarray:
        """Calculate error propagation for filled pixels"""
        if method not in ['nearest', 'biharmonic']:
            raise ValueError("method must be one of: 'nearest', 'biharmonic'")
        
        if method == 'nearest':
            if 'footprint' not in kwargs:
                raise ValueError("footprint is required for nearest method")
            
            def error_propagation(values):
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    return np.nan
                # No longer divide by n_valid since we use median filling
                # Conservative estimate: use root mean square of errors in neighborhood
                return np.sqrt(np.nanmean(values[valid_mask]**2))
            
            filled_errors = self._iterative_fill(
                error, mask, error_propagation, kwargs['footprint']
            )
            
        else:  # 'biharmonic'
            # Use maximum error from mask boundary as the error for filled region
            boundary_mask = ndimage.binary_dilation(mask) & ~mask
            max_boundary_error = np.max(error[boundary_mask])
            
            filled_errors = error.copy()
            filled_errors[mask] = max_boundary_error
        
        return filled_errors
    
    def by_median_map(self, image: np.ndarray, mask: np.ndarray,
                     median_map: np.ndarray) -> np.ndarray:
        """Fill using median map"""
        filled = image.copy()
        filled[mask] = median_map[mask]
        return filled
    
    def by_tile_median(self, image: np.ndarray, tile_size: int, 
                       mask: np.ndarray = None, return_template: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        将图像按 tile 分块，每块用该块的中位数值填充。

        参数：
            image (np.ndarray): 2D 图像数组，可包含 NaN。
            tile_size (int): tile 大小（正方形 tile）。

        返回：
            np.ndarray: 用 tile 中值填充的图像。
        """
        ny, nx = image.shape
        template = image.copy()
        if mask is None:
            mask = np.zeros_like(image, dtype=bool)
        template[mask] = np.nan

        for y0 in range(0, ny, tile_size):
            for x0 in range(0, nx, tile_size):
                y1 = min(y0 + tile_size, ny)
                x1 = min(x0 + tile_size, nx)
                tile = template[y0:y1, x0:x1]

                # 提取非 NaN 有效值
                valid_values = tile[~np.isnan(tile)]
                if valid_values.size > 0:
                    median_val = np.median(valid_values)
                    template[y0:y1, x0:x1] = median_val
                    # 可选：tile 中的 NaN 保持为 NaN，不做替换
                else:
                    # 整个 tile 都是 NaN，保持为 NaN
                    continue
                
        filled = image.copy()
        filled[mask] = template[mask]
        if return_template:
            return filled, template
        else:
            return filled

class BackgroundCleaner:
    """High-level interface combining StarIdentifier and PixelFiller"""
    
    def __init__(self):
        self.identifier = StarIdentifier()
        self.filler = PixelFiller()
    
    def process_single_image(self, image: np.ndarray,
                           identify_method: str = 'sigma_clip',
                           fill_method: str = 'neighbors',
                           **kwargs) -> np.ndarray:
        """Complete workflow for processing a single image"""
        # Choose identification method
        if identify_method == 'sigma_clip':
            mask = self.identifier.by_sigma_clip(image, **kwargs)
        elif identify_method == 'manual':
            mask = self.identifier.by_manual(image, **kwargs)
        else:
            raise ValueError(f"Unsupported identify method: {identify_method}")
        
        # Choose filling method
        if fill_method == 'neighbors':
            cleaned = self.filler.by_neighbors(image, mask, **kwargs)
        elif fill_method == 'ring':
            cleaned = self.filler.by_ring(image, mask, **kwargs)
        elif fill_method == 'median_map':
            cleaned = self.filler.by_median_map(image, mask, **kwargs)
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")
        
        return cleaned
    
    def process_image_pair(self, image1: np.ndarray, image2: np.ndarray,
                          threshold: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
        """Complete workflow for processing an image pair"""
        mask_pos, mask_neg = self.identifier.by_comparison(image1, image2, threshold)
        cleaned1, cleaned2 = self.filler.by_comparison(image1, image2, mask_pos, mask_neg)
        return cleaned1, cleaned2

class StarCleaner:
    @staticmethod
    def identify_stars(img: np.ndarray, 
                       identify_method: str = 'sigma_clip',
                       identify_paras: Union[Dict[str, Any], None] = None,
                       focus_region: Union[np.ndarray, PixelRegion, None] = None,
                       exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                       identified_mask: np.ndarray = None):
        """
        'sigma_clip':
        identify_paras = {'sigma': float = 3, 'maxiters': int = 3, 'tile_size': int = None, 'area_size': None, \
            'expand_shrink_paras': {'radius': float = 1, 'method': 'expand' or 'shrink', 'speend': 'normal' or 'fast'}}
        'manual':
        identify_paras = {'row_range': tuple(int, int) = None, 'col_range': tuple(int, int) = None, \
                        'vmin': float = 0, 'vmax': float = 10, 
                        'region_plot': Union[PixelRegion, List[PixelRegion]] = None}
        """
        # exclude region
        exclude_region = get_exclude_region(img.shape, focus_region, exclude_region)
        # identify
        star_identifier = StarIdentifier()
        if identify_method == 'sigma_clip':
            if identify_paras is None:
                identify_paras = {'sigma': 3, 'maxiters': 3, 'tile_size': None, 'area_size': None, \
                                  'expand_shrink_paras': None}
            sigma = identify_paras.get('sigma', 3)
            maxiters = identify_paras.get('maxiters', 3)
            tile_size = identify_paras.get('tile_size', None)
            area_size = identify_paras.get('area_size', None)
            expand_shrink_paras = identify_paras.get('expand_shrink_paras', None)
            mask = star_identifier.by_sigma_clip(img, sigma=sigma, maxiters=maxiters, 
                                                 exclude_region=exclude_region, tile_size=tile_size,
                                                 area_size=area_size, expand_shrink_paras=expand_shrink_paras)
        elif identify_method == 'manual':
            if identify_paras is None:
                identify_paras = {'target_coord': (1000, 1000), 'radius': 50, 
                                  'vmin': 0, 'vmax': 10, 'save_path': None, 'region_plot': None}
            target_coord = identify_paras.get('target_coord', (1000, 1000))
            radius = identify_paras.get('radius', 50)
            col_range = (target_coord[0]-radius, target_coord[0]+radius)
            row_range = (target_coord[1]-radius, target_coord[1]+radius)
            vmin = identify_paras.get('vmin', 0)
            vmax = identify_paras.get('vmax', 10)
            save_path = identify_paras.get('save_path', None)
            region_plot = identify_paras.get('region_plot', None)
            mask = star_identifier.by_manual(img, row_range=row_range, col_range=col_range, vmin=vmin, vmax=vmax, save_path=save_path, region_plot=region_plot)
        else:
            raise ValueError(f"Currently unsupported identify method: {identify_method}")
        if identified_mask is not None:
            mask = identified_mask | mask
        return mask
    
    @staticmethod
    def fill_stars(img: np.ndarray, mask: np.ndarray,
                   fill_method: str = 'neighbors',
                   fill_paras: Union[Dict[str, Any], None] = None,
                   focus_region: Union[np.ndarray, PixelRegion, None] = None,
                   exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                   ):
        """
        'neighbors':
        fill_paras = {'radius': int = 4, 'method': str = 'nearest', 'mean_or_median': str = 'mean'} # 'median_filter', 'uniform_filter'
        'tile_median':
        fill_paras = {'tile_size': int = 40}
        'rings':
        fill_paras = {'center': tuple(int, int), 'step': float, 'method': str = 'median', \
                      'start': float = None, 'end': float = None} # 'median', 'mean'
        """
        exclude_region = get_exclude_region(img.shape, focus_region, exclude_region)
        if exclude_region is not None:
            mask = mask & ~exclude_region
        star_filler = PixelFiller()
        if fill_method == 'neighbors':
            if fill_paras is None:
                fill_paras = {'radius': 4, 'method': 'nearest', 'mean_or_median': 'median'}
            radius = fill_paras.get('radius', 4)
            method = fill_paras.get('method', 'nearest') # 'median_filter'
            mean_or_median = fill_paras.get('mean_or_median', 'mean')
            filled = star_filler.by_neighbors(img, mask=mask, radius=radius, method=method, mean_or_median=mean_or_median)
        elif fill_method == 'tile_median':
            if fill_paras is None:
                fill_paras = {'tile_size': 40}
            tile_size = fill_paras.get('tile_size', 40)
            filled = star_filler.by_tile_median(img, tile_size=tile_size, mask=mask)
        elif fill_method == 'rings':
            if fill_paras is None:
                fill_paras = {'center': (1000, 1000), 'step': 2, 'method': 'median', 'start': None, 'end': None}
            center = fill_paras.get('center', (1000, 1000))
            step = fill_paras.get('step', 2)
            method = fill_paras.get('method', 'median') # 'median', 'mean'
            start = fill_paras.get('start', None)
            end = fill_paras.get('end', None)
            filled = star_filler.by_rings(img, mask=mask, center=center, step=step, method=method, start=start, end=end)
        elif fill_method == 'median_map':
            if fill_paras is None:
                fill_paras = {'median_map': None}
            median_map = fill_paras.get('median_map', None)
            filled = star_filler.by_median_map(img, mask=mask, median_map=median_map)
        else:
            raise ValueError(f"Currently unsupported fill method: {fill_method}")
        return filled
    

def save_starmask(img_path: Union[str, Path], 
                  img_extension: str,
                  star_mask_steps: List[str],
                  identify_method: str = 'sigma_clip', # 'sigma_clip', 'manual'
                  identify_paras: Dict[str, Any] = None,
                  focus_region: Union[np.ndarray, PixelRegion, None] = None, 
                  exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                  plot: bool = False,
                  plot_vrange: Union[Tuple[float, float], None] = None,
                  save: bool = False,
                  verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    'sigma_clip':
    identify_paras = {'sigma': float = 3, 'maxiters': int = 3, 'tile_size': int = None, 'area_size': None, \
            'expand_shrink_paras': {'radius': float = 1, 'method': 'expand' or 'shrink', 'speend': 'normal' or 'fast'}
            }
    'manual':
    identify_paras = {'target_coord': tuple(int, int) = (1000, 1000), 'radius': int = 50, \
                    'vmin': float = 0, 'vmax': float = 10, 
                    'region_plot': Union[PixelRegion, List[PixelRegion]] = None}
    """
    with fits.open(img_path, mode='readonly') as hdul:
        img = hdul[img_extension].data.copy()
        has_starmask = 'STARMASK' in hdul
    mask = StarCleaner.identify_stars(img, identify_method=identify_method, \
                                      identify_paras=identify_paras, \
                                        focus_region=focus_region, \
                                            exclude_region=exclude_region)
    if verbose:
        print(f'{np.sum(mask)} star pixels identified')
    
    if plot:
        if plot_vrange is None:
            plot_vrange = (np.min(img), np.max(img)*0.1)
        inspector = MaskInspector(img, mask)
        inspector.show_comparison(vmin=plot_vrange[0], vmax=plot_vrange[1])
    if save:
        star_mask_steps.append(identify_method)
        if has_starmask:
            with fits.open(img_path, mode='update') as hdul:
                hdu_starmask = hdul['STARMASK']
                hdu_starmask.data = hdu_starmask.data | mask.astype(int)
                hdul[0].header['MASKHIST'] = ','.join(star_mask_steps)
                hdul[0].header['STARMASK'] = (True, 'Star mask applied')
            if verbose:
                print(f'STARMASK extension updated in {img_path}')
        else:
            with fits.open(img_path, mode="readonly") as hdul:
                hdus = [hdu.copy() for hdu in hdul]
            ext_num = len(hdus)
            hdu_starmask = fits.ImageHDU(mask.astype(int), name='STARMASK')
            hdus.append(hdu_starmask)
            hdus[0].header[f'EXT{ext_num}NAME'] = ('STARMASK', f'Name of extension {ext_num}')
            hdus[0].header['MASKHIST'] = ','.join(star_mask_steps)
            hdus[0].header['STARMASK'] = (True, 'Star mask applied')
            fits.HDUList(hdus).writeto(img_path, overwrite=True)
            if verbose:
                print(f'STARMASK extension created in {img_path}')
    return star_mask_steps, mask

def delete_starmask(img_path, verbose: bool = False):
    with fits.open(img_path, mode='readonly') as hdul:
        # 找到 STARMASK 扩展的索引
        starmask_idx = None
        for i, hdu in enumerate(hdul):
            if hdu.name == 'STARMASK':
                starmask_idx = i
                break
        if starmask_idx is None:
            if verbose:
                print(f'STARMASK extension not found in {img_path}')
            return
        hdus = [hdu.copy() for hdu in hdul if hdu.name != 'STARMASK']
    if f'EXT{starmask_idx}NAME' in hdus[0].header:
        del hdus[0].header[f'EXT{starmask_idx}NAME']
    if 'MASKHIST' in hdus[0].header:
        del hdus[0].header['MASKHIST']
    if 'STARMASK' in hdus[0].header:
        del hdus[0].header['STARMASK']
    fits.HDUList(hdus).writeto(img_path, overwrite=True)
    if verbose:
        print(f'STARMASK extension deleted in {img_path}')

def save_filled(img_path: Union[str, Path],
                img_extension: str,
                star_fill_steps: List[str],
                mask: Union[np.ndarray, None] = None, 
                fill_method: str = 'neighbors', 
                fill_paras: Dict[str, Any] = None,
                focus_region: Union[np.ndarray, PixelRegion, None] = None,
                exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                plot: bool = False,
                plot_vrange: Union[Tuple[float, float], None] = None,
                save: bool = False,
                verbose: bool = False):
    with fits.open(img_path, mode='readonly') as hdul:
        img = hdul[img_extension].data.copy()
        has_starmask = 'STARMASK' in hdul
        if has_starmask and mask is None:
            mask = hdul['STARMASK'].data.astype(bool)
        elif not has_starmask and mask is None:
            raise ValueError("No starmask found")
    filled = StarCleaner.fill_stars(img, mask, 
                                    fill_method=fill_method, 
                                    fill_paras=fill_paras,
                                    focus_region=focus_region,
                                    exclude_region=exclude_region)
    if verbose:
        print('Bad pixel fillment finished.')
    if plot:
        if plot_vrange is None:
            plot_vrange = (np.min(img), np.max(img)*0.1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img, vmin=plot_vrange[0], vmax=plot_vrange[1], origin='lower')
        ax1.set_title('Original Image')
        ax2.imshow(filled, vmin=plot_vrange[0], vmax=plot_vrange[1], origin='lower')
        ax2.set_title('Filled Image')
        plt.show(block=True)
    if save:
        star_fill_steps.append(fill_method)
        with fits.open(img_path, mode='update') as hdul:
            hdul[img_extension].data = filled
            hdul[0].header['STARFILL'] = (True, 'Star fill applied')
            hdul[0].header['FILLHIST'] = ','.join(star_fill_steps)
        if verbose:
            print(f'{img_extension} extension updated in {img_path}')
    return star_fill_steps