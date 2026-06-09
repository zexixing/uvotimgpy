# uvotimgpy

`uvotimgpy` is a small Python package for processing and analyzing astronomical
images, with tools aimed at Swift/UVOT and HST workflows.

The package includes utilities for FITS image handling, image alignment and
cropping, aperture photometry, star masking and cleaning, radial-profile work,
and simple UVOT observation planning helpers.

## Installation

From the repository root:

```bash
pip install -e .
```

The code uses common astronomy Python packages such as `astropy`,
`astroquery`, `matplotlib`, `numpy`, and `regions`.

## Package Layout

- `uvotimgpy/base`: common file, math, region, and visualization helpers
- `uvotimgpy/utils`: image and spectrum operations
- `uvotimgpy/uvot_image`: UVOT image correction, morphology, and star cleaning
- `uvotimgpy/uvot_analysis`: aperture photometry and activity analysis
- `uvotimgpy/uvot_plan`: planning and exposure-time tools
- `uvotimgpy/uvot_file`: file organization and FITS I/O helpers

## Example

```python
from uvotimgpy.config import paths

print(paths.package_uvotimgpy)
```

## Notes

Image coordinates are often converted between DS9-style coordinates and Python
array coordinates. FITS files are handled with `astropy.io.fits`, and some
interactive tools are designed for use in a Jupyter or matplotlib Qt session.
