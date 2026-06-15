from typing import Tuple, Optional, List
import numpy as np
from scipy.interpolate import interp1d
from uvotimgpy.utils.image_operation import calc_radial_profile, GeometryMap, profile_to_image
import matplotlib.pyplot as plt
class ImageEnhancer:
    @staticmethod
    def by_azimuthal_median(image: np.ndarray,
                            center: Tuple[float, float],
                            step: float = 1.0,
                            edge_list: Optional[List[float]] = None,
                            image_err: Optional[np.ndarray] = None,
                            bad_pixel_mask: Optional[np.ndarray] = None,
                            start: Optional[float] = None,
                            end: Optional[float] = None,
                            method: str = 'division',
                            median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
                            power: float = 0.5,
                            return_model_image: bool = False,
                            return_radial_profile: bool = False) -> np.ndarray:
        """
        Enhance a comet image using the azimuthal median profile.
        method: 'division' or 'subtraction'
        TODO: add angle limit to get radial_profile
        """
        # Get the radial profile
        radial_profile = calc_radial_profile(
            image=image,
            center=center,
            step=step,
            edge_list=edge_list,
            image_err=image_err,
            bad_pixel_mask=bad_pixel_mask,
            start=start,
            end=end,
            method='median',
            median_err_params=median_err_params,
            power=power
        )
        
        if image_err is not None:
            radii, values, _ = radial_profile
        else:
            radii, values = radial_profile
        
        value_center = image[center[1], center[0]] # center[0]:col, center[1]:row
        # radii = np.insert(radii, 0, 0) # Insert 0 at the first position of radii
        # values = np.insert(values, 0, value_center) # Insert value_center at the first position of values
        
        # Create the distance image
        distance_map = GeometryMap(image, center).get_distance_map()

        # Create the 2D model image using an interpolation function
        model_image = profile_to_image(radii, values, distance_map=distance_map, fill_value=np.nan, start_r=0, start_value=value_center)
        if return_model_image:
            return model_image
        ## Use DistanceMap to get the index map
        #distance_map = DistanceMap(image, center)
        #index_map = distance_map.get_index_map(step)

        ## Create a lookup table whose length matches the maximum index_map value
        #lut = np.full(index_map.max() + 1, np.nan)
        ## Fill values into the corresponding index positions
        #lut[0:len(values)] = values

        ## Use the index map to get model_image
        #model_image = lut[index_map]
        
        # Compute the enhanced image; set locations with zero model values to NaN
        if method == 'division':
            #enhanced_image = np.where(model_image <= 0, np.nan, image / model_image)
            enhanced_image = image / model_image
        elif method == 'subtraction':
            enhanced_image = image - model_image
        else:
            raise ValueError(f"Unsupported method: {method}")

        if not return_radial_profile:
            return enhanced_image
        else:
            return enhanced_image, radii, values
    
    def by_one_over_rho(image: np.ndarray,
                      center: Tuple[float, float],
                      return_model_image: bool = False,
                      ) -> np.ndarray:
        r = DistanceMap(image, center).get_distance_map()
        one_over_rho = 1 / r
        enhanced_image = image / one_over_rho
        if return_model_image:
            return one_over_rho
        return enhanced_image

def bin_profile(radii, values, errors=None, bin_width=10):
    import numpy as np
    
    r_min, r_max = radii.min(), radii.max()
    bins = np.arange(r_min, r_max + bin_width, bin_width)
    bin_idx = np.digitize(radii, bins)

    br, bv = [], []
    if errors is not None:
        be = []

    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue

        r_bin = radii[mask]
        v_bin = values[mask]

        if errors is not None:
            e_bin = errors[mask]
            w = 1 / e_bin**2
            val = np.sum(w * v_bin) / np.sum(w)
            err = np.sqrt(1 / np.sum(w))
        else:
            val = np.mean(v_bin)

        br.append(np.mean(r_bin))
        bv.append(val)
        if errors is not None:
            be.append(err)

    if errors is not None:
        return np.array(br), np.array(bv), np.array(be)
    else:
        return np.array(br), np.array(bv)
