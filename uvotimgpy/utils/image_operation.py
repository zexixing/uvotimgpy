from astropy.io import fits
import numpy as np
#from scipy.ndimage import shift, rotate
from skimage.transform import rotate
from typing import List, Tuple, Union, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from regions import CircleAnnulusPixelRegion, CirclePixelRegion, PixCoord
from scipy.interpolate import interp1d
from uvotimgpy.base.math_tools import ErrorPropagation
from uvotimgpy.base.region import mask_image, RegionStatistics, expand_shrink_region, create_sector_region, create_sector_region_from_map
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.nddata import block_reduce
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter, gaussian_filter
warnings.filterwarnings('ignore', category=FITSFixedWarning)

class DS9Converter:
    def __init__(self):
        """
        Initialize the object.
        """
        pass
    @staticmethod
    def round_to_int(x) -> int:
        return int(np.floor(x + 0.5))
    
    @staticmethod
    def ds9_to_coords(ds9_x: Union[float, int], 
                      ds9_y: Union[float, int], 
                      to_int: bool = True) -> Union[Tuple[int, int, int, int], 
                                                   Tuple[float, float, float, float]]:
        """
        Convert DS9 coordinates to Python coordinates.
        
        Parameters
        ----------
        ds9_x, ds9_y : float or int
            Source coordinates in DS9, starting from 1, with integer pixel centers.
            ds9_x: horizontal coordinate, the first number.
            ds9_y: vertical coordinate, the second number.
        to_int : bool, optional
            Whether to convert the result to integers; default is True.
            
        Returns
        -------
        python_column, python_row : int or float
            Indices in the Python array, starting from 0.
            python_column corresponds to ds9_x, the second array index.
            python_row corresponds to ds9_y, the first array index.
        """
        if to_int:
            # DS9 coordinate range [m-0.5, m+0.5) corresponds to integer m
            ds9_out_x = DS9Converter.round_to_int(ds9_x)
            ds9_out_y = DS9Converter.round_to_int(ds9_y)
        else:
            ds9_out_x = ds9_x
            ds9_out_y = ds9_y
        
        # Python indices start from 0
        python_column = ds9_out_x - 1
        python_row = ds9_out_y - 1
        
        return python_column, python_row
    
    @staticmethod
    def coords_to_ds9(python_column: Union[float, int],
                      python_row: Union[float, int], 
                      to_int: bool = True) -> Union[Tuple[int, int, int, int], 
                                                   Tuple[float, float, float, float]]:
        """
        Convert Python array coordinates to DS9 coordinates.
        
        Parameters
        ----------
        python_column, python_row : float or int
            Indices in the Python array, starting from 0.
            python_column corresponds to ds9_x, the second array index.
            python_row corresponds to ds9_y, the first array index.
        to_int : bool, optional
            Whether to convert the result to integers; default is True.
            
        Returns
        -------
        ds9_x, ds9_y : int or float
            Coordinates in DS9, starting from 1.
        """
        # DS9 coordinates start from 1
        ds9_x = python_column + 1
        ds9_y = python_row + 1
        
        if to_int:
            ds9_x = DS9Converter.round_to_int(ds9_x)
            ds9_y = DS9Converter.round_to_int(ds9_y)
            
        return ds9_x, ds9_y

def exposure_mask_with_nan(image: np.ndarray, 
                           exposure_map: np.ndarray, 
                           exposure_threshold: Optional[float] = None) -> np.ndarray:
    """
    Replace image values with NaN where the exposure map is below 0 or below/equal to the threshold.
    """
    if image.shape != exposure_map.shape:
        raise ValueError("image and exposure_map must have the same shape")
    if exposure_threshold is None:
        image[exposure_map <= 0] = np.nan
    else:
        image[exposure_map < exposure_threshold] = np.nan
    return image

def rescale_images(images: Union[np.ndarray, List[np.ndarray]], 
                  current_scales: Union[float, List[float]], 
                  new_scale: Optional[float] = None,
                  target_coords: Optional[List[Tuple[float, float]]] = None,
                  headers: Optional[List[fits.Header]] = None) -> Tuple:
    # TODO: check if this is correct
    """
    Rescale images to a new pixel scale.
    
    Parameters
    ----------
    images : single image array or list of image arrays
    current_scales : current pixel scale or list of scales
    new_scale : new pixel scale; if None, use the largest scale
    target_coords : list of source coordinates in each image [(x1,y1), (x2,y2),...]
    headers : list of FITS headers
    
    Returns
    -------
    rescaled_images, new_coords, updated_headers if headers are provided
    """
    # Ensure consistent input format
    if not isinstance(images, list):
        images = [images]
        current_scales = [current_scales]
        target_coords = [target_coords] if target_coords is not None else None
    
    # If no new scale is specified, use the largest scale
    if new_scale is None:
        new_scale = max(current_scales)
    
    rescaled_images = []
    new_coords = []
    updated_headers = []
    
    for i, (img, scale) in enumerate(zip(images, current_scales)):
        # Calculate the scaling factor
        factor = scale / new_scale
        
        if abs(factor - round(factor)) < 1e-10:  # Integer-ratio relation
            factor = int(round(factor))
            if factor > 1:  # Need to shrink
                new_img = img[::factor, ::factor]
            elif factor < 1:  # Need to enlarge
                factor = abs(factor)
                shape = np.array(img.shape) * factor
                new_img = np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
            else:
                new_img = img.copy()
        else:
            # Non-integer ratio; other interpolation methods can be used
            new_img = img  # Other interpolation methods can be implemented here
            
        rescaled_images.append(new_img)
        
        # Update coordinates
        if target_coords:
            x, y = target_coords[i]
            new_coords.append((x/factor, y/factor))
            
        # Update header
        if headers:
            header = headers[i].copy()
            header['PIXSCALE'] = new_scale
            updated_headers.append(header)
    
    if headers:
        return rescaled_images, new_coords, updated_headers
    elif target_coords:
        return rescaled_images, new_coords
    else:
        return rescaled_images

def rotate_image(image: np.ndarray, 
                rotate_center: Tuple[Union[float, int], Union[float, int]], 
                angle: float,
                fill_value: Union[float, None] = np.nan,
                image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Rotate an image around the source position.
    
    Parameters
    ----------
    image : input image
    rotate_center : source coordinates (col, row)
    angle : rotation angle in degrees
    fill_value : fill value
    """
    if image.dtype.byteorder not in ('=', '<'):
        image = image.byteswap().view(image.dtype.newbyteorder('<'))
    rotated_img =  rotate(image, 
                          -angle,
                          center=rotate_center,
                          preserve_range=True,
                          mode='constant',
                          cval=fill_value,    # Specify the fill value
                          clip=True)
    if image_err is None:
        return rotated_img
    else:
        rotated_err = rotate_image(image = image_err, target_coord = rotate_center, angle = angle, fill_value = fill_value)
        return rotated_img, rotated_err

def crop_image(image: np.ndarray, 
              old_target_coord: Tuple[Union[float, int], Union[float, int]], 
              new_target_coord: Tuple[Union[float, int], Union[float, int]],
              fill_value: Union[float, None] = np.nan,
              image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Crop an image around the source position.
    
    Parameters
    ----------
    image : input image
    target_coord : source coordinates in the original image (column, row)
    new_target_coord : desired source coordinates in the new image (column, row)
    fill_value : fill value
    """
    col, row = old_target_coord
    new_col, new_row = new_target_coord

    if isinstance(new_col, float):
        new_col = round(new_col)
    if isinstance(new_row, float):
        new_row = round(new_row)
    if isinstance(col, float):
        col = round(col)
    if isinstance(row, float):
        row = round(row)
    # Calculate the new image size
    new_size = (2 * new_row + 1, 2 * new_col + 1)
    if isinstance(fill_value, int):
        fill_value = float(fill_value)
    new_image = np.full(new_size, fill_value)
    
    # Calculate the crop range
    start_col = col - new_col
    start_row = row - new_row
    
    # Copy the valid region
    valid_region = np.s_[
        max(0, start_row):min(image.shape[0], start_row + new_size[0]),
        max(0, start_col):min(image.shape[1], start_col + new_size[1])
    ]
    new_valid_region = np.s_[
        max(0, -start_row):min(new_size[0], image.shape[0]-start_row),
        max(0, -start_col):min(new_size[1], image.shape[1]-start_col)
    ]
    
    new_image[new_valid_region] = image[valid_region]

    if image_err is None:
        return new_image
    else:
        cropped_err = crop_image(image_err, old_target_coord, new_target_coord, fill_value=fill_value)
        return new_image, cropped_err


def align_images(images: List[np.ndarray], 
                target_coords: List[Tuple[Union[float, int], Union[float, int]]],
                new_target_coord: Tuple[Union[float, int], Union[float, int]],
                fill_value: Union[float, None] = np.nan,
                image_err: Optional[List[np.ndarray]] = None) ->  Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Align a series of images.
    """
    if image_err is None:
        return [crop_image(img, coord, new_target_coord, fill_value) 
                for img, coord in zip(images, target_coords)]
    else:
        aligned_images = []
        aligned_errs = []
        for image, coord, err in zip(images, target_coords, image_err):
            aligned_image, aligned_err = crop_image(image, coord, new_target_coord, fill_value, err)
            aligned_images.append(aligned_image)
            aligned_errs.append(aligned_err)
        return aligned_images, aligned_errs

def images_by_exposure(images: List[np.ndarray], 
                        exposure_list: Optional[Union[List[np.ndarray], List[float]]],
                        method: str = 'divide') -> List[np.ndarray]:
    """
    Divide images by exposure maps.
    """
    if all(isinstance(expo, np.ndarray) for expo in exposure_list):
        for img, expo in zip(images, exposure_list):
            img[(expo / np.max(expo)) < 0.99] = np.nan
    #exposure_stack = np.sum(exposure_list, axis=0)
    if method == 'divide':
        images = [img / expo for img, expo in zip(images, exposure_list)]
    elif method == 'multiply':
        images = [img * expo for img, expo in zip(images, exposure_list)]
    else:
        raise ValueError("method must be 'divide' or 'multiply'")
    return images
        

def stack_images(images: List[np.ndarray], 
                 method: str = 'median',
                 image_err: Optional[List[np.ndarray]] = None,
                 median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
                 verbose: bool = False,
                 input_unit: Optional[str] = None,
                 exposure_list: Optional[Union[List[np.ndarray], List[float]]] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stack images.
    
    Parameters
    ----------
    images : image list
    method : 'median', 'mean', or 'sum'
    median_err_params: dict, optional
        median_err_params for ErrorPropagation.median: method ('mean' or 'std'), mask (True/False)
    """
    if method not in ['median', 'mean', 'sum']:
        raise ValueError("method must be 'median' or 'mean' or 'sum'")
    if method == 'mean' or method == 'sum':
        if verbose:
            warnings.warn("Some pixels may be not well exposed, please check the exposure map with sum_exposure_map().")
    all_nan = np.all(np.isnan(images), axis=0)
    if method == 'mean':
        if input_unit == 'count' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='divide')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='divide')
        if image_err is not None:
            mean_image, mean_error = ErrorPropagation.mean(images, image_err, axis=0, ignore_nan=True)
            mean_image[all_nan] = np.nan
            mean_error[all_nan] = np.nan
            return mean_image, mean_error
        else:
            result = np.nanmean(images, axis=0)
            result[all_nan] = np.nan
            return result
    elif method == 'median':
        if input_unit == 'count' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='divide')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='divide')
        if image_err is not None:
            median_image, median_error = ErrorPropagation.median(images, image_err, axis=0, ignore_nan=True,
                                                                 **median_err_params)
            median_image[all_nan] = np.nan
            median_error[all_nan] = np.nan
            return median_image, median_error
        else:
            result = np.nanmedian(images, axis=0)
            result[all_nan] = np.nan
            return result
    elif method == 'sum':
        if input_unit == 'count/s' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='multiply')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='multiply')
        if image_err is not None:
            sum_image, sum_error = ErrorPropagation.sum(images, image_err, axis=0, ignore_nan=True)
            sum_image[all_nan] = np.nan
            sum_error[all_nan] = np.nan
            return sum_image, sum_error
        else:
            result = np.nansum(images, axis=0)
            result[all_nan] = np.nan
            return result

def shrink_valid_image(image: np.ndarray, shrink_pixels: int = 2, mask: Optional[np.ndarray] = None, speed: str = 'normal') -> np.ndarray:
    """
    Shrink valid regions in the image inward by the specified number of pixels,
    i.e. expand the original np.nan region or mask outward by shrink_pixels pixels.
    
    Parameters:
        image (np.ndarray): input 2D image array; invalid regions must be marked with np.nan
        shrink_pixels (int): number of pixels to shrink, i.e. number of pixels to expand NaN by

    Returns:
        np.ndarray: processed image with the expanded NaN region.
    """
    if image.ndim != 2:
        # raise ValueError("输入必须是二维图像")
        raise ValueError("Input must be a 2D image")

    # Build the NaN-region mask
    if mask is None:
        shrink_mask = np.isnan(image)
    else:
        shrink_mask = mask

    # Expand the NaN region, eroding the valid region inward
    expanded_mask = expand_shrink_region(shrink_mask, radius=shrink_pixels, method='expand', speed=speed)

    # Create a new image copy
    new_image = image.copy()
    new_image[expanded_mask] = np.nan

    return new_image

def sum_exposure_map(exposure_maps: Optional[List[np.ndarray]] = None,
                     images: Optional[List[np.ndarray]] = None,
                     exposures: Optional[List[float]] = None) -> np.ndarray:
    """
    Stack exposure maps.
    """
   # if isinstance(exposure_maps, type(None)):
    if exposure_maps is None:
        exposure_maps = []
        for image, exposure in zip(images, exposures):
            image_copy = image.copy()
            image_copy[~np.isnan(image_copy)] = 1. # TODO: check pixels with values as 0
            exposure_maps.append(image_copy * exposure)
    #else: # for Swift which has exposure map fits files
    #    pass
    summed_exposure_map = np.nansum(exposure_maps, axis=0)
    summed_exposure_map = np.nan_to_num(summed_exposure_map, 0)
    return summed_exposure_map
    

class ImageDistanceCalculator:
    @staticmethod
    def calc_distance(coords1, coords2, wcs=None, scale=None):
        """Calculate the distance between two points."""
        # Calculate the pixel distance directly
        pixel_dist = np.sqrt((coords2[0] - coords1[0])**2 + 
                           (coords2[1] - coords1[1])**2)
        
        if wcs is None:
            if scale is None:
                return pixel_dist
            else:
                return pixel_dist * scale
            
        # Use WCS conversion
        sky1 = wcs.pixel_to_world(coords1[0], coords1[1])
        sky2 = wcs.pixel_to_world(coords2[0], coords2[1])
        return sky1.separation(sky2)
        
    @staticmethod
    def from_edges(image, coords, distance_method='max', return_coords=False, wcs=None, scale=None):
        """Calculate the distance to the image edge.
        
        Args:
            max_distance: True returns the maximum distance; False returns the minimum distance.
        """
        n_rows, n_cols = image.shape
        col, row = coords
    
        edges = [
            (col, 0),
            (col, n_rows),
            (0, row),
            (n_cols, row)
        ]

        distances = [(ImageDistanceCalculator.calc_distance(coords, edge), edge) for edge in edges]
        if distance_method == 'max':
            dist, edge = max(distances)
        elif distance_method == 'min':
            dist, edge = min(distances)

        if return_coords:
            return edge
        return ImageDistanceCalculator.calc_distance(coords, edge, wcs, scale)
        
    @staticmethod
    def from_corners(image, coords, distance_method='max', return_coords=False, scale=None, wcs=None):
        """Calculate the distance to image corners.
        
        Args:
            max_distance: True returns the maximum distance; False returns the minimum distance.
        """
        n_rows, n_cols = image.shape
        col, row = coords

        corners = [
            (0, 0),
            (0, n_rows),
            (n_cols, 0),
            (n_cols, n_rows)
        ]

        distances = [(ImageDistanceCalculator.calc_distance(coords, corner), corner) for corner in corners]
        if distance_method == 'max':
            dist, corner = max(distances)
        elif distance_method == 'min':
            dist, corner = min(distances)

        if return_coords:
            return corner
        return ImageDistanceCalculator.calc_distance(coords, corner, wcs, scale)

    @staticmethod
    def max_distance_from_valid_pixels(image: np.ndarray, 
                                     coords: Tuple[Union[float, int], Union[float, int]], 
                                     ) -> float:
        """Calculate the maximum distance from non-NaN pixels in the image to a specified point.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
        coords : tuple
            Reference point coordinates (col, row).
        wcs : object, optional
            WCS object used for sky-coordinate conversion.
        scale : float, optional
            Pixel scale used to convert pixel distance to physical distance.
            
        Returns
        -------
        float
            Maximum distance, in pixels or in the unit converted by scale/WCS.
        """
        # Create coordinate grids
        rows, cols = np.indices(image.shape)
        
        # Calculate the distance from each pixel to the reference point
        col_diff = cols - coords[0]
        row_diff = rows - coords[1]
        distances = np.sqrt(col_diff**2 + row_diff**2)
        
        # Set distances corresponding to NaN pixels to NaN
        distances[np.isnan(image)] = np.nan
        
        # Calculate the maximum distance
        max_dist = np.nanmax(distances)
            
        return max_dist

class GeometryMap:
    """Class for handling distances from image pixels to a specified center."""
    
    def __init__(self, image: np.ndarray, center: tuple):
        #self.image = image
        #self.center_col, self.center_row = center
        #
        ## Calculate the distance map directly during initialization
        #rows, cols = np.indices(self.image.shape)
        #self.dist_map = np.sqrt(
        #    (cols - self.center_col)**2 + 
        #    (rows - self.center_row)**2
        #)
        height, width = image.shape

        # Create the pixel coordinate grid
        cols = np.arange(width)
        rows = np.arange(height)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Calculate the distance from each pixel to the center
        self.center_col, self.center_row = center
        self.dcol = col_grid - self.center_col
        self.drow = row_grid - self.center_row
        
    def get_distance_map(self) -> np.ndarray:
        """Calculate the distance from each pixel to the center."""
        return np.sqrt(self.dcol**2 + self.drow**2)
    
    def get_range_mask(self, inner_radius: float, outer_radius: float) -> np.ndarray:
        """
        Get the pixel mask within the specified distance range.
        
        Parameters
        ----------
        inner_radius : float
            Inner radius.
        outer_radius : float
            Outer radius.
            
        Returns
        -------
        np.ndarray
            Boolean mask; pixels within the specified range are True.
        """
        self.dist_map = np.sqrt(self.dcol**2 + self.drow**2)
        return (self.dist_map >= inner_radius) & (self.dist_map < outer_radius)
    
    def get_index_map(self, step) -> np.ndarray:
        """Get the distance index map."""
        #index_map = np.round(self.dist_map / step).astype(int)
        #index_map = np.maximum(index_map, 1)
        self.dist_map = np.sqrt(self.dcol**2 + self.drow**2)
        index_map = np.floor(self.dist_map / step).astype(int)
        return index_map

    def get_angle_map(self) -> np.ndarray:
        """Get each pixel's angle relative to the center."""
        angle_rad = np.arctan2(self.drow, self.dcol)
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 270) % 360 # 0 degrees = up, 90 degrees = left, 180 degrees = down, 270 degrees = right

def build_edge_list(start: float,
                    end: float,
                    step: float = 1.0,
                    power: float = 0.0):
    """
    Vectorized bin construction using
        width_i = step * (1 + i^beta) / 2

    power = 0: linear (width = step always)
    power > 0: increasing widths
    """
    if end <= start:
        raise ValueError("end must > start")
    if step <= 0:
        raise ValueError("step must > 0")

    full_span = end - start

    # --- (1) Roughly estimate the maximum number of bins needed, an upper limit for the linear case ---
    # Because width_i >= step/2
    max_bins = int(np.ceil(full_span / (step / 2)))  
    # max_bins is usually larger than the actual valid count, but this does not hurt performance

    # --- (2) Generate all widths for i = 0..max_bins at once ---
    i = np.arange(max_bins, dtype=float)[1:]

    #if beta == 0.0:
    #    widths = np.full_like(i, step)
    #else:
    widths = step * (1 + i**power) / 2

    # --- (3) Cumulative sum ---
    cumsum = widths.cumsum()

    # --- (4) Find the range where all bins can be fully included ---
    valid = cumsum <= full_span
    if not np.any(valid):
        # If not even one bin fits, fall back to a single bin
        return np.array([start, end], dtype=float)

    last = np.where(valid)[0][-1]  # Last valid bin index

    # --- (5) Build edge_list ---
    widths_used = widths[: last + 1]
    edge_offsets = np.concatenate(([0.0], widths_used.cumsum()))
    edge_list = start + edge_offsets

    return edge_list

#def calc_radial_profile(image: np.ndarray, 
#                       center: tuple, 
#                       step: float,
#                       edge_list: Optional[List[float]] = None,
#                       image_err: Optional[np.ndarray] = None,
#                       bad_pixel_mask: Optional[np.ndarray] = None,
#                       sector_pa: Optional[float] = None,
#                       sector_span: Optional[float] = None,
#                       start: Optional[float] = None,
#                       end: Optional[float] = None,
#                       method: str = 'median',
#                       median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
#                       power: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#    """Calculate the radial profile and its errors.
#
#    Parameters
#    ----------
#    image : np.ndarray
#        Input image
#    center : tuple
#        Center coordinates (col, row)
#    step : float
#        Annulus width
#    edge_list : List[float], optional
#        Edge list
#    image_err : np.ndarray, optional
#        Image error array
#    bad_pixel_mask : np.ndarray, optional
#        Bad-pixel mask; True indicates masked pixels
#    start : float, optional
#        Starting radius; default is 0
#    end : float, optional
#        Ending radius; default is the distance from the image center to the corner
#    method : str
#        Calculation method: 'median', 'mean', or 'max'
#
#    Returns
#    -------
#    radii : np.ndarray
#        Radius array
#    values : np.ndarray
#        Corresponding profile values
#    errors : np.ndarray, optional
#        If image_err is provided, return the corresponding error values
#    """
#    # Process input image and mask
#    if sector_pa is not None and sector_span is not None:
#        sector_region = create_sector_region(center, sector_pa, sector_span, image.shape, radius=end)
#        out_sector_mask = ~sector_region
#        if bad_pixel_mask is None:
#            bad_pixel_mask = out_sector_mask
#        else:
#            bad_pixel_mask = bad_pixel_mask | out_sector_mask
#    image = mask_image(image, bad_pixel_mask)
#    #image_err = mask_image(image_err, bad_pixel_mask) if not isinstance(image_err, type(None)) else None
#    image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
#    # Process center coordinates
#    if isinstance(center, PixCoord):
#        center_coord = center
#    else:
#        center_coord = PixCoord(x=center[0], y=center[1])
#    
#    # Set start and end radii
#    start = start if start is not None else 0
#    if end is None:
#        end = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
#    
#    # Initialize result lists
#    if edge_list is None and step is not None:
#        #edge_list = np.arange(start, end+step, step)
#        edge_list = np.asarray(build_edge_list(start, end, step, power=power))
#    radii = []
#    values = []
#    errors = [] if image_err is not None else None
#
#    # Calculate the value at each radius
#    for i, r in enumerate(edge_list[:-1]):
#        # Create region
#        r_next = edge_list[i+1]
#        if r == 0:
#            region = CirclePixelRegion(
#                center=center_coord,
#                radius=r_next
#            )
#            radii.append(r)
#        else:
#            region = CircleAnnulusPixelRegion(
#                center=center_coord,
#                inner_radius=r,
#                outer_radius=r_next
#            )
#            radii.append((r+r_next)/2)
#
#        # Get valid pixel values within the region
#        mask = region.to_mask()
#        region_pixels = mask.get_values(image)
#        valid_pixels = region_pixels[~np.isnan(region_pixels)]
#
#        if len(valid_pixels) == 0:
#            values.append(np.nan)
#            if image_err is not None:
#                errors.append(np.nan)
#            continue
#
#        if image_err is not None:
#            region_errors = mask.get_values(image_err)
#            valid_errors = region_errors[~np.isnan(region_pixels)]
#
#        if method == 'median':
#            if image_err is not None:
#                value, error = ErrorPropagation.median(valid_pixels, valid_errors, axis=None, ignore_nan=True, 
#                                                       **median_err_params)
#                values.append(value)
#                errors.append(error)
#            else:
#                value = np.nanmedian(valid_pixels)
#                values.append(value)
#        elif method == 'mean':
#            if image_err is not None:
#                value, error = ErrorPropagation.mean(valid_pixels, valid_errors, axis=None, ignore_nan=True)
#                values.append(value)
#                errors.append(error)
#            else:
#                value = np.nanmean(valid_pixels)
#                values.append(value)
#
#    # Convert to numpy arrays and return results
#    radii = np.array(radii)
#    values = np.array(values)
#    if image_err is not None:
#        errors = np.array(errors)
#        return radii, values, errors
#    return radii, values
    
def calc_radial_profile(image: np.ndarray, 
                       center: tuple, 
                       step: float,
                       edge_list: Optional[List[float]] = None,
                       image_err: Optional[np.ndarray] = None,
                       bad_pixel_mask: Optional[np.ndarray] = None,
                       sector_pa: Optional[float] = None,
                       sector_span: Optional[float] = None,
                       start: Optional[float] = None,
                       end: Optional[float] = None,
                       method: str = 'median',
                       median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
                       power: float = 1.0,
                       distance_map: Optional[np.ndarray] = None,
                       angle_map: Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Calculate the radial profile and its errors.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    center : tuple
        Center coordinates (col, row).
    step : float
        Annulus width.
    edge_list : List[float], optional
        Edge list.
    image_err : np.ndarray, optional
        Image error array.
    bad_pixel_mask : np.ndarray, optional
        Bad-pixel mask; True indicates masked pixels.
    start : float, optional
        Starting radius; default is 0.
    end : float, optional
        Ending radius; default is the distance from the image center to the corner.
    method : str
        Calculation method: 'median', 'mean', or 'max'.
    power = 0: linear (width = step always)

    Returns
    -------
    radii : np.ndarray
        Radius array.
    values : np.ndarray
        Corresponding profile values.
    errors : np.ndarray, optional
        If image_err is provided, return the corresponding error values.
    """
    # Process input image and mask
    if distance_map is None or angle_map is None:
        geometry_mapper = GeometryMap(image, center)
        if distance_map is None:
            distance_map = geometry_mapper.get_distance_map()
        if angle_map is None:
            angle_map = geometry_mapper.get_angle_map()
    if sector_pa is not None and sector_span is not None:
        #sector_region = create_sector_region(center, sector_pa, sector_span, image.shape, radius=end, )
        sector_region = create_sector_region_from_map(distance_map, angle_map, sector_pa, sector_span, radius=end)
        out_sector_mask = ~sector_region
        if bad_pixel_mask is None:
            bad_pixel_mask = out_sector_mask
        else:
            bad_pixel_mask = bad_pixel_mask | out_sector_mask
    value_center = image[center[1], center[0]]
    if image_err is not None:
        error_center = image_err[center[1], center[0]]
    else:
        error_center = None
    image = mask_image(image, bad_pixel_mask)
    #image_err = mask_image(image_err, bad_pixel_mask) if not isinstance(image_err, type(None)) else None
    image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
    # Process center coordinates
    if isinstance(center, PixCoord):
        center_coord = center
    else:
        center_coord = PixCoord(x=center[0], y=center[1])
    
    # Set start and end radii
    start = start if start is not None else 0
    if end is None:
        end = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
    
    # Initialize result lists
    if edge_list is None and step is not None:
        #edge_list = np.arange(start, end+step, step)
        edge_list = np.asarray(build_edge_list(start, end, step, power=power))

    n_bin = len(edge_list) - 1
    # Calculate the representative radius for each bin; default is the midpoint, but force 0 for the innermost bin if it starts at 0
    radii = 0.5 * (edge_list[:-1] + edge_list[1:])
    if edge_list[0] == 0:
        radii[0] = 0.0

    # Flatten to 1D and build a valid-pixel mask
    radius_flat = distance_map.ravel()
    image_flat = image.ravel()
    finite = np.isfinite(image_flat)
    # Only include pixels whose radii are within the edge_list range
    r_min, r_max = edge_list[0], edge_list[-1]
    radius_in = (radius_flat >= r_min) & (radius_flat <= r_max)
    valid = finite & radius_in
    if image_err is not None:
        image_err_flat = image_err.ravel()
        finite_err = np.isfinite(image_err_flat)
        #valid &= finite_err
    else:
        image_err_flat = None
    if not np.any(valid):
        values = np.full(n_bin, np.nan, dtype=float)
        errors = np.full(n_bin, np.nan, dtype=float) if image_err is not None else None
        if errors is not None:
            return radii, values, errors
        else:
            return radii, values

    r_flat = radius_flat[valid]
    v_flat = image_flat[valid]
    if image_err_flat is not None:
        e_flat = image_err_flat[valid]
    else:
        e_flat = None

    # Assign pixels to bins by radius
    bin_idx = np.digitize(r_flat, edge_list) - 1   # 0 to n_bin-1
    # Defensive clipping for extreme boundaries
    good_bins = (bin_idx >= 0) & (bin_idx < n_bin)
    r_flat = r_flat[good_bins]
    v_flat = v_flat[good_bins]
    if e_flat is not None:
        e_flat = e_flat[good_bins]
    bin_idx = bin_idx[good_bins]
    # Initialize outputs
    values = np.full(n_bin, np.nan, dtype=float)
    errors = np.full(n_bin, np.nan, dtype=float) if e_flat is not None else None
    if edge_list[0] == 0:
        values[0] = value_center
        if e_flat is not None:
            errors[0] = error_center
    if edge_list[0] == 0:
        loop_range = range(1, n_bin)
    else:
        loop_range = range(n_bin)
    # Calculate each bin one by one (median/mean/max); this is still a Python loop
    for i in loop_range:
        sel = (bin_idx == i)
        if not np.any(sel):
            continue
        pix = v_flat[sel]
        if e_flat is not None:
            errs = e_flat[sel]
        else:
            errs = None
#        if radii[i] == 0:
#            print('hi')
#            values[i] = value_center
#            if errs is not None:
#                errors[i] = error_center
        if method == 'median':
            if errs is not None:
                # Keep the original error propagation logic
                value, error = ErrorPropagation.median(
                    pix, errs, axis=None, ignore_nan=True, **median_err_params
                )
                values[i] = value
                errors[i] = error
            else:
                values[i] = np.nanmedian(pix)
        elif method == 'mean':
            if errs is not None:
                value, error = ErrorPropagation.mean(
                    pix, errs, axis=None, ignore_nan=True
                )
                values[i] = value
                errors[i] = error
            else:
                values[i] = np.nanmean(pix)
        elif method == 'min':
            values[i] = np.nanmin(pix)
        else:
            raise ValueError(f"Unknown method '{method}', must be 'median' or 'mean'.")
    if errors is not None:
        return radii, values, errors
    else:
        return radii, values

def calc_radial_profiles_sectors(
    image: np.ndarray,
    center: tuple,
    step: float,
    sector_pa_list: Sequence[float],
    sector_span: Union[float, Sequence[float]],
    edge_list: Optional[Sequence[float]] = None,
    image_err: Optional[np.ndarray] = None,
    bad_pixel_mask: Optional[np.ndarray] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    method: str = "median",
    median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
    power: float = 1.0,
    verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Calculate a radial profile for each sector_pa.

    Parameters
    ----------
    sector_pa_list : sequence of float
        Position angle for each sector.
    sector_span : float or sequence of float
        If float/int, all sectors use the same span.
        If sequence, its length must match sector_pa_list.
    Other parameters:
        Same as calc_radial_profile; if edge_list is None, it is generated once internally.

    Returns
    -------
    radii : (n_rad,) ndarray
    values : (n_sector, n_rad) ndarray
        Each row corresponds to one sector profile.
    errors : (n_sector, n_rad) ndarray or None
    """
    sector_pa_list = np.asarray(sector_pa_list, dtype=float)
    n_sector = len(sector_pa_list)

    # Normalize sector_span
    if np.isscalar(sector_span):
        sector_span_list = np.full(n_sector, float(sector_span))
    else:
        sector_span_list = np.asarray(sector_span, dtype=float)
        if len(sector_span_list) != n_sector:
            # raise ValueError("sector_span 长度必须与 sector_pa_list 一致。")
            raise ValueError("sector_span length must match sector_pa_list.")
    # Precompute radius_map / theta_map shared by all sectors
    geometry_mapper = GeometryMap(image, center)
    distance_map = geometry_mapper.get_distance_map()
    angle_map = geometry_mapper.get_angle_map()
    if bad_pixel_mask is not None:
        image = mask_image(image, bad_pixel_mask)
        image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
    # If edge_list is not provided, generate one from the full image for all sectors
    if edge_list is None:
        # Use the build_edge_list helper defined above
        start_val = 0.0 if start is None else start
        # Temporarily estimate end using one max_distance calculation
        if end is None:
            end_val = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
        else:
            end_val = end
        edge_list = build_edge_list(start_val, end_val, step, power=power)

    edge_list = np.asarray(edge_list)

    profile_dict = {}

    for pa, span in zip(sector_pa_list, sector_span_list):
        result = calc_radial_profile(
            image=image,
            center=center,
            step=step,                  # No longer used to generate edge_list, but kept for API consistency
            edge_list=edge_list,        # Fix the same edge_list for all sectors
            image_err=image_err,
            bad_pixel_mask=None,
            sector_pa=pa,
            sector_span=span,
            start=start,
            end=end,
            method=method,
            median_err_params=median_err_params,
            power=power,
            distance_map=distance_map,
            angle_map=angle_map,
        )
        if image_err is not None:
            radii, vals, errs = result
            profile_dict[pa] = {'radii': radii, 'values': vals, 'errors': errs}
        else:
            radii, vals = result
            profile_dict[pa] = {'radii': radii, 'values': vals}
        if verbose:
            print(f'sector {pa} done')
    return profile_dict


def bin_image(image: np.ndarray, 
              block_size: Union[int, Tuple[int, int]], 
              image_err: Optional[np.ndarray] = None,
              method: str = 'mean',
              median_err_params: Optional[dict] = {'method':'mean', 'mask':True}) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply binning to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    block_size : int or tuple of int
        Binning block size. If an integer, use the same size in both directions.
        If a tuple, specify the block size separately in the (col, row) directions.
    method : str, optional
        Function used for binning; default is np.nanmean.
    image_err : np.ndarray, optional
        Image error array.
        
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        Binned image; if image_err is provided, also return the binned error.
    """
    if method == 'mean':
        func = np.nanmean
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.mean(a_err, a_err, axis=axis, ignore_nan=True)
            return error
        
    elif method == 'median':
        func = np.nanmedian
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.median(a_err, a_err, axis=axis, ignore_nan=True, **median_err_params)
            return error
    elif method == 'sum':
        func = np.nansum
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.sum(a_err, a_err, axis=axis, ignore_nan=True)
            return error
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    if isinstance(block_size, int):
        block_size = (block_size, block_size)
        
    binned_image = block_reduce(image, block_size, func=func)
    
    if image_err is None:
        return binned_image
    else:
        if ErrorPropagation.check_consistency(image, image_err):
            binner_err = block_reduce(image_err, block_size, func=error_func)
            return binned_image, binner_err
        else:
            raise ValueError("image and image_err must have consistent shapes and NaN positions")

def bin_coordinate(pixel_coord, block_size):
    """
    Map original image pixel coordinates pixel_coord to binned image coordinates after block_reduce.

    Parameters
    ----------
    pixel_coord: (col, row)
    block_size : tuple(int, int) or int
        Block size in each direction, as (bcol, brow).

    Returns
    ----------
    (colb, rowb) : tuple(int, int)
        Pixel coordinates in the binned image.
    """
    col, row = pixel_coord
    if isinstance(block_size, int):
        bcol, brow = block_size, block_size
    else:
        bcol, brow = block_size
    rowb = row // brow
    colb = col // bcol
    return (colb, rowb)

def get_original_coordinate_range(pixel_coord, block_size):
    """
    Given binned image coordinates (yb, xb), return the corresponding original-image pixel ranges.

    Parameters
    ----------
    pixel_coord : (col, row)
        Binned image coordinates.
    block_size : tuple(int, int) or int
        Block size in each direction, as (bcol, brow).

    Returns
    ----------
    (colrange, rowrange) : tuple(range, range)
        Corresponding column and row ranges in the original image.
    """
    if isinstance(block_size, int):
        bcol, brow = block_size, block_size
    else:
        bcol, brow = block_size
    col, row = pixel_coord
    colrange = range(col * bcol, (col + 1) * bcol)
    rowrange = range(row * brow, (row + 1) * brow)
    return colrange, rowrange

def tile_image(image: np.ndarray, tile_size: int, func=np.median, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Split the image into tiles and fill each tile with its median value.
    Parameters
    ----------
    image : np.ndarray
        Input image.
    tile_size : int
        Tile size.
    func : callable
        Fill function, e.g. np.median or np.mean.
    mask : np.ndarray, optional
        Mask.
    """
    image_tiled = image.copy()
    ny, nx = image.shape
    if mask is not None:
        image_tiled[mask] = np.nan
    for y0 in range(0, ny, tile_size):
        for x0 in range(0, nx, tile_size):
            y1 = min(y0 + tile_size, ny)
            x1 = min(x0 + tile_size, nx)
            tile = image_tiled[y0:y1, x0:x1]
            
            # Extract valid non-NaN values
            valid_values = tile[~np.isnan(tile)]
            if valid_values.size > 0:
                filled_val = func(valid_values)
                image_tiled[y0:y1, x0:x1] = filled_val
                # Optional: keep NaNs in the tile as NaN without replacement
            else:
                # Entire tile is NaN; keep it as NaN
                continue
    return image_tiled

def smooth_image(image: np.ndarray, size: int = 3, method: str = 'gaussian_filter',
                 image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Smooth an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    kernel_size : int
        Kernel size.
    method : str
        Smoothing method: 'convolve', 'gaussian_filter', 'boxcar', or 'fft'.
    """
    if image_err is not None:
        consistency = ErrorPropagation.check_consistency(image, image_err)
        if not consistency:
            raise ValueError("image and image_err must have consistent shapes and NaN positions")
    if method == 'convolve':
        kernel = Gaussian2DKernel(size)
        smoothed_image = convolve(image, kernel)
        mask = ~np.isfinite(image)
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            smoothed_err = np.sqrt(convolve(image_err**2, kernel**2))
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
    elif method == 'gaussian_filter':
        smoothed_image = gaussian_filter(image, sigma=size)
        if image_err is None:
            return smoothed_image
        else:
            smoothed_err = gaussian_filter(image_err, sigma=size)
            return smoothed_image, smoothed_err
    elif method == 'boxcar':
        mask = ~np.isfinite(image)
        image[mask] = 0.0
        smoothed_image = uniform_filter(image, size=size)
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            image_err[mask] = 0.0
            smoothed_err = uniform_filter(image_err, size=size)
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
    elif method == 'fft':
        mask = ~np.isfinite(image)
        image[mask] = 0.0
        kernel = Gaussian2DKernel(size)
        kernel = kernel.array.astype(np.float32)
        smoothed_image = fftconvolve(image, kernel, mode='same')
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            image_err[mask] = 0.0
            smoothed_err = fftconvolve(image_err, kernel, mode='same')
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
        

def profile_to_image(profile_r, profile_value, distance_map=None, radius=None, fill_value=np.nan,
                     start_r = None, start_value = None):
    """
    Convert a radial profile to an image.
    fill_value: np.nan, 0, 'extrapolate'
    """
    if start_r is not None:

        if start_r < profile_r[0]:
            profile_r = np.insert(profile_r, 0, start_r)
            profile_value = np.insert(profile_value, 0, start_value)
        elif start_r > profile_r[0]:
            profile_value = profile_value[profile_r >= start_r]
            profile_r = profile_r[profile_r >= start_r]
        else:
            pass
    profile_interp = interp1d(profile_r, profile_value, 
                              bounds_error=False, 
                              fill_value=fill_value)
    if (distance_map is None) and (radius is not None):
        radius = int(radius)
        center = (radius, radius)
        shape = (2*radius+1, 2*radius+1)
        image = np.zeros(shape)
        distance_calculator = GeometryMap(image, center)
        distance_map = distance_calculator.get_distance_map()
    elif distance_map is not None:
        pass
    elif (distance_map is not None) and (radius is not None):
        raise ValueError("distance_map and radius cannot be provided at the same time")
    else:
        raise ValueError("distance_map or radius must be provided")
    image = profile_interp(distance_map)
    return image

def center_of_image(image: np.ndarray):
    H = len(image)
    W = len(image[0])
    center_col = (W - 1) / 2
    center_row = (H - 1) / 2
    return center_col, center_row

def upscale_mean_fill(
    image: np.ndarray,
    n: int,
    conserve_flux: bool = False,) -> np.ndarray:
    """
    Enlarge a 2D image by an integer factor n using local-mean block filling.

    Each original pixel becomes an n×n block filled with the mean of its
    surrounding neighborhood (here: the pixel value itself, so equivalent
    to constant block replication). This matches typical astronomical
    area-style upscaling.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    n : int
        Upscaling factor (n >= 1).
    conserve_flux : bool, optional
        If True, divide the result by n^2 so that total sum of pixels
        is conserved.

    Returns
    -------
    upscaled : 2D ndarray
        Enlarged image of shape (ny*n, nx*n).
    """
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer")

    img = np.asarray(image, dtype=float)

    if img.ndim != 2:
        raise ValueError("image must be 2D")

    # Block replication (fast, no loops)
    up = np.repeat(np.repeat(img, n, axis=0), n, axis=1)

    if conserve_flux:
        up /= n * n

    return up
