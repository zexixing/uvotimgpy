# uvotimgpy

`uvotimgpy` is a work-in-progress Python package for processing and analyzing
small body images, with tools mainly aimed at Swift/UVOT.


## Package Layout

- `uvotimgpy/base`: common file, math, region, and visualization helpers
- `uvotimgpy/utils`: image and spectrum operations
- `uvotimgpy/uvot_image`: UVOT image correction, morphology, and star cleaning
- `uvotimgpy/uvot_analysis`: aperture photometry and activity analysis
- `uvotimgpy/pipeline`: basic and advanced image-processing/analyzing pipelines
- `uvotimgpy/uvot_plan`: observation planning and exposure-time tools
- `uvotimgpy/uvot_file`: observation log / file organization and FITS I/O helpers
