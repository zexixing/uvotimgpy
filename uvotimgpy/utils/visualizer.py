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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left panel: masked pixels
        masked_img = np.copy(self.image)
        masked_img[~self.mask] = np.nan
        im1 = ax1.imshow(masked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(im1, ax=ax1, label='Pixel Value')
        ax1.set_title('Masked Pixels')
        
        # Right panel: unmasked pixels
        unmasked_img = np.copy(self.image)
        unmasked_img[self.mask] = np.nan
        im2 = ax2.imshow(unmasked_img, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.colorbar(im2, ax=ax2, label='Pixel Value')
        ax2.set_title('Unmasked Pixels')
        
        plt.tight_layout()
        plt.show()