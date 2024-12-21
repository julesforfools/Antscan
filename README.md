# Antscan
A collection of scripts, programs, and resources to recreate and perform the post-processing of Antscan data.
Post-processing was undertaken to aid in dataset optimization and visualization, including:
- Merging scans of specimens imaged in several height steps
- Cropping out background via deep learning specimen segmentation
- Whole-body STL Mesh generation via deep learning specimen segmentation
- 2D-image series generation
- Downsampled 2D-image series generation
- Automated mesh-to-image rendering

Access to DNN for segmentations prior to public access:
https://filesender2.oist.jp/filesender/?s=download&token=43c079c7-2e9d-4cfc-975e-11db53bb623d

## Test Data
The test_data folder emulates the file/folder structure on which postprocessing was executed.
We offer all raw, intermediary, and final formats to allow recreation of the workflow at any stage.

## Merging and Cropping Scans
General usage:
python3 -u antscan_register_merge_crop.py <path-to-scan-folder> <z-shift> <path-to-processed-folder> <path-to-dnn> <path-to-biomedisa> &> <log-filename>.log &

Test data usage (example folder structure Windows):
python antscan_register_merge_crop.py C:/Users/Julian/git/Antscan/test_data/5x 165 C:/Users/Julian/git/Antscan/test_data/processed/5x/ G:/3d_workdir/ant_sf/6_antscan_dnn/antscan_2024-08-15_b_augment.h5
Note : cd into directory with antscan_register_merge_crop.py first
Note : replace folder names when recreating
Runtime of script on Testdata: 33 seconds (Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz, 128 GB RAM, Nvidia Quadro RTX 6000)

Files generated:
- specimens.txt
- intermediary_result.nii
- 16-02.tif (merged and cropped scan)
- specimens_processed_0-0.txt (merging and cropping metadata)

## Generating STL Meshes and 2D-Image Series
General usage:
nohup python3 -u antscan_make_biomedisa_STL_PNG.py <path-to-processed-folder> &> <log-filename>.log &

Test data usage (example folder structure Windows):
python antscan_make_biomedisa_STL_PNG.py C:/Users/Julian/git/Antscan/test_data/processed/5x/ G:/3d_workdir/ant_sf/6_antscan_dnn/antscan_2024-08-15_b_augment.h5 
Note : cd into directory with antscan_make_biomedisa_STL_PNG.py first
Note : replace folder names when recreating
Runtime of script on Testdata: 32 seconds (Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz, 128 GB RAM, Nvidia Quadro RTX 6000)

Files generated:
- 16-02 folder (2D image series)
- 16-02.stl (3D Mesh)
- specimens_STLPNG.txt (STL metadata)

## Render Images of STL Meshes 
General usage:
nohup python3 -u antscan_screenshots_main_v4.py <path-to-processed-folder> &> <log-filename>.log &
Note: Requires presence of 'antscan_paraview_screenshot_v3.py' in folder

Test data usage (example folder structure Windows):
python antscan_screenshots_main_v4.py C:/Users/Julian/git/Antscan/test_data/processed/5x/
Note : cd into directory with antscan_screenshots_main_v4.py first
Note : replace folder names when recreating
Runtime of script on Testdata: 13 seconds (Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz, 128 GB RAM, Nvidia Quadro RTX 6000)

File generated:
- 16-02.png (STL render image)

## Downsample 2D Image Series
General usage:
nohup python3 -u antscan_resize_slices.py &> <log-filename>.log &

Test data usage (example folder structure Windows):
python antscan_resize_slices.py C:/Users/Julian/git/Antscan/test_data/processed/5x/
Note : cd into directory with antscan_resize_slices.py first
Note : replace folder names when recreating
Runtime of script on Testdata: 2 seconds (Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz, 128 GB RAM, Nvidia Quadro RTX 6000)

Files generated:
- 16-02_lowres folder (no files generated as testing image (202x202 px) already below downsampling threshold (400x400 px))
