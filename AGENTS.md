# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

uvotimgpy is a Python package for processing and analyzing astronomical images, specifically designed for UVOT (Ultraviolet/Optical Telescope) and HST (Hubble Space Telescope) data. The package provides tools for image operations, star cleaning, aperture photometry, and spectrum analysis.

## Package Structure

- `uvotimgpy/` - Main package directory
  - `base/` - Core functionality and utilities
    - `file_io.py` - File input/output operations
    - `math_tools.py` - Mathematical utilities
    - `region.py` - Region selection and manipulation
    - `visualizer.py` - Visualization tools including MaskInspector
  - `utils/` - Utility modules
    - `image_operation.py` - Image processing (rotation, alignment, cropping)
    - `spectrum_operation.py` - Spectrum analysis tools
    - `filters.py` - Filter-related operations
  - `uvot_analysis/` - UVOT-specific analysis tools
    - `aperture_photometry.py` - Photometric measurements
    - `activity.py` - Activity analysis
    - `planner.py` - Observation planning
  - `uvot_file/` - File organization and management
    - `file_organization.py` - Data file handling
  - `uvot_image/` - Image processing specialized for UVOT
    - `star_cleaner.py` - Star identification and removal (StarIdentifier, PixelFiller, BackgroundCleaner)
    - `morphology.py` - Morphological analysis
  - `config.py` - Project configuration and paths (ProjectPaths class)
  - `query.py` - Star catalog queries (StarCoordinateQuery)
  - `hst_handler.py` - HST data handling

## Key Components

### Image Processing Pipeline
- **DS9Converter**: Coordinate conversion between DS9 and image coordinates
- **StarIdentifier**: Multiple methods for star identification:
  - `by_sigma_clip()` - Statistical detection
  - `by_catalog()` - Catalog-based (GSC, UCAC4)
  - `by_manual()` - Interactive manual selection
- **PixelFiller**: Fill masked regions using various methods (nearest neighbors, biharmonic)
- **BackgroundCleaner**: Process image pairs for background subtraction

### Analysis Tools
- **MaskInspector**: Visualization tool for inspecting masked regions
- **RegionSelector**: Interactive region selection for photometry
- **calc_radial_profile()**: Radial profile analysis

## Development Commands

### Installation and Setup
```bash
pip install -e .  # Install in development mode
```

### Testing
Tests are located in `tests/` directory as Jupyter notebooks:
- `test_image.ipynb` - Image processing tests
- `test_utils.ipynb` - Utility function tests
- `test_error.ipynb` - Error handling tests

Run tests using Jupyter:
```bash
jupyter notebook tests/
```

### Data Paths
The package uses a centralized path configuration in `config.py`. The ProjectPaths class manages:
- Work directory: `/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork`
- Data paths for HST and Swift observations
- Package and project organization

## Dependencies

Key dependencies include:
- astropy (FITS handling, WCS, coordinates)
- astroquery (catalog queries via Vizier, SIMBAD)
- matplotlib (visualization)
- numpy (numerical operations)
- regions (astronomical regions)

## Important Notes

- Image coordinates follow DS9 convention (1-indexed) and are converted to Python arrays (0-indexed)
- FITS files are handled with astropy.io.fits
- Interactive plotting uses matplotlib with Qt backend (`%matplotlib qt`)
- Error propagation is supported in pixel filling operations
- Catalog queries support GSC and UCAC4 star catalogs

