# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:19:07 2023

@author: Julian Katzke, Philipp D. Loesel
"""

############################################################################################
############################### Setup ######################################################
############################################################################################

import os # for files and directories
import sys # to simply run biomedisa
import numpy as np
import scipy.ndimage # to perform processes after DNN segmentation
import time
import psutil # to monitor memory usage
import warnings
import ast
#import imageio
from skimage import io

### Setup Biomedisa (version 24.7.1 or higher)
# load libraries
#sys.path.append(path_to_biomedisa)
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.deeplearning import deep_learning
from biomedisa.features.active_contour import activeContour
from biomedisa.mesh import save_mesh

print(sys.argv[0:])
#print(len(sys.argv))
#sys.argv is a list in Python, which contains the command-line arguments passed to the script.
#sys.argv[0] is the script, sys.argv[1] is the source file directory,
#sys.argv[2] is the target file directory
#sys.argv[3] is the dnn

############################################################################################
############################### Functions ##################################################
############################################################################################

def absolute_folder_paths(directory):
    ''' Return Paths of all folders (scans) in parent folder'''
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path)]

def absolute_tif_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.name.endswith('.tif')]

def rmislands(seg, thresh = 0.01):

    '''
    This function takes a segmentation and removes small components.
    This function takes in an image in numpy.
    Included here is a largest island selection to filter the noise in the segmentation.
    The default threshold should is 10%.
    The value assumed for segmented voxels here is '1'
    '''

    ### Largest islands segmentation with scipy to kick out small components
    ############################################################### Start a timer
    li_start_time = time.time()
    ################################################################################
    mask, num_features = scipy.ndimage.label(seg, output = np.int32)
    component_sizes = np.bincount(mask.ravel())
    largest_component_size = component_sizes[1:].max()
    # Define the minimum size threshold (1% of the largest component)
    min_size_threshold = thresh * largest_component_size
    # Create an output image to store the filtered components
    result = np.zeros_like(mask)
    # Keep only the components larger than the size threshold
    for component_label, size in enumerate(component_sizes):
        if component_label == 0:  # Background
            continue
        if size >= min_size_threshold:
            result[mask == component_label] = 1 # Use one from Biomedisa segmentation
    # Convert back to uint8
    mask = result.astype(np.uint8)
    ################################################################ End the timer
    li_end_time = time.time()
    # Calculate the execution time in seconds
    li_execution_time = li_end_time - li_start_time
    # Print the execution time
    print(" Small Islands Removal time:", li_execution_time, "seconds")
    ####################################################################################

    return(mask)


############################################################################################
############################### Script #####################################################
############################################################################################
parent_folder = sys.argv[1]
target_folder = os.path.abspath(sys.argv[1])

specimens = absolute_tif_paths(parent_folder) # get folders/specimens
specimens = sorted(specimens) # sort folders alphabetically!

print("Total number of specimens:", len(specimens))
print(specimens)


############################################################### Start a timer
start_time = time.time()
################################################################################

#############################################################################################
#################### Make STL files and PNG previews for all files ##########################
#############################################################################################
for specimen in specimens:

    ############################################################### Start a timer
    spec_start_time = time.time()
    ################################################################################

    spec_name = os.path.basename(specimen)
    print("\n Next specimen: ", spec_name)
    print("Look at this folder name: ", specimen)
    print("Look at this output file name: ", os.path.join(target_folder, spec_name.replace('.tif','.stl')))

    if os.path.isdir(specimen.replace('.tif','')) and os.path.isfile(specimen.replace('.tif','.stl')):
        print(spec_name, " STL and PNGs already exist.")
    else:
        # load data
        main_image, img_header, img_ext = load_data(specimen,
            return_extension=True)

        ### Print Memory usage
        # Get the current process ID
        pid = psutil.Process()
        # Get the memory usage in bytes
        memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
        # Print the memory usage in gigabytes
        print("Memory usage :", memory_usage, "GB")
        ###

        ################ PNG Stack ###############
        if not os.path.isdir(specimen.replace('.tif','')):
            # make png directory
            os.makedirs(specimen.replace('.tif',''))

            # Directory to save the PNG files
            output_dir = specimen.replace('.tif','')

            ############################################################### Start a timer
            png_start_time = time.time()
            ################################################################################
            # Save each slice as a separate PNG file
            for i in range(main_image.shape[0]):
                slice_image = main_image[i, :, :]  # Extract the i-th slice along the z-axis
                output_path = os.path.join(output_dir, f'slice_{i:04d}.png')
                io.imsave(output_path, slice_image)
            ################################################################ End the timer
            png_end_time = time.time()
            # Calculate the execution time in seconds
            png_execution_time = png_end_time - png_start_time
            # Print the execution time
            print(" PNG export time:", png_execution_time, "seconds")
            ####################################################################################

        ################ STL ###############
        if not os.path.isfile(os.path.join(target_folder, spec_name.replace('.tif','.stl'))):
            ########################################################
            ########## Set Voxel Size and threshold value ##########
            # Use the image path to feel out the voxel size and adjust using the helper function
            if '/10x/' in specimen:
                x_res, y_res, z_res = 1.22,1.22,1.22
            elif '/5x/' in specimen:
                x_res, y_res, z_res = 2.44,2.44,2.44
            elif '/2x/' in specimen:
                x_res, y_res, z_res = 6.11,6.11,6.11
            elif '/GAGA_10x/' in specimen:
                x_res, y_res, z_res = 1.22,1.22,1.22
            elif '/GAGA_5x/' in specimen:
                x_res, y_res, z_res = 2.44,2.44,2.44
            elif '/CSOSZ_5x/' in specimen:
                x_res, y_res, z_res = 2.44,2.44,2.44
            elif '/GAGA_2x/' in specimen:
                x_res, y_res, z_res = 6.11,6.11,6.11
            elif '/CT-Lab/' in specimen:
                x_res, y_res, z_res = 8.20,8.20,8.20
            #Windows tests:
            if '\\5x\\' in specimen:
                x_res, y_res, z_res = 2.44,2.44,2.44
            elif '\\10x\\' in specimen:
                x_res, y_res, z_res = 1.22,1.22,1.22
            elif '\\2x\\' in specimen:
                x_res, y_res, z_res = 6.11,6.11,6.11
            #########################################################
            ### Segment Surface ###
            # deep learning
            #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Assign GPU 2 to the main script
            results = deep_learning(main_image, predict=True, img_header=img_header,
                path_to_model=sys.argv[2], img_extension=img_ext, batch_size = 24)

            # Filter smallest Islands
            segmentation1 = rmislands(results['regular'], 0.05)
            # Get unique values and their counts
            unique_values, counts = np.unique(segmentation1, return_counts=True)

            ### Set poly reduction if resulting mesh is expected to exceed 3 Million faces ###
            # Print unique values and their counts
            for value, count in zip(unique_values, counts):
                print(f"Value: {value}, Count: {count}")
            # If the segmentation exceeds 200 Million voxels, we get a very large mesh ~>300MB, then we reduce
            if counts[1] > 150000000:
                pr = 1-(150000000/counts[1])
                print("Large Mesh expected, reduction value will be set to:", pr, "that is he data set is reduced to ", (1-pr)*100, "% of its original size")
            else:
                pr = 0

            # create stl
            #print(os.path.join(target_folder, spec_name.replace('.tif','.stl')))
            #save_mesh(os.path.join(target_folder, spec_name.replace('.tif','_NEW_RAW.stl')), segmentation1, x_res, y_res, z_res, poly_reduction = pr, smoothing_iterations= 0)

            # Refine segmentation
            segmentation = activeContour(main_image, segmentation1, simple=False)
            #segmentation = activeContour(main_image, results['regular'])
            del results #deallocate object to save memory

            # Invert Numpy array to get rid of mirroring
            ###########################################################################
            ############################################################### Start a timer
            flip_start_time = time.time()
            ################################################################################
            inverted_segmentation = segmentation[::-1, :, :]
            ############################################################### End the timer
            flip_end_time = time.time()
            #Calculate the execution time in seconds
            flip_execution_time = flip_end_time - flip_start_time
            # Print the execution time
            print(" Mirroring time:", flip_execution_time, "seconds")
            ####################################################################################

            ###########################################################################
            ############################################################### Start a timer
            stl_start_time = time.time()
            ################################################################################
            # screate stl with from create_mesh import CreateSTL
            save_mesh(os.path.join(target_folder, spec_name.replace('.tif','.stl')), inverted_segmentation, x_res, y_res, z_res, poly_reduction = pr, smoothing_iterations= 15)
            with open(os.path.join(target_folder,"specimens_STLPNG.txt"), 'a') as fp:
                fp.write("%s\n" % (specimen+ ' ' + str(x_res) + ' ' + str(pr)))
            ############################################################### End the timer
            stl_end_time = time.time()
            #Calculate the execution time in seconds
            stl_execution_time = stl_end_time - stl_start_time
            # Print the execution time
            print(" STL export time:", stl_execution_time, "seconds")
            ####################################################################################
            del segmentation
            del segmentation1
        del main_image
    ################################################################ End the timer
    spec_end_time = time.time()
    # Calculate the execution time in seconds
    spec_execution_time = spec_end_time - spec_start_time
    # Print the execution time
    print(" Total time for specimen: ", spec_name, " ", spec_execution_time/60, "minutes")
    ####################################################################################




################################################################ End the timer
end_time = time.time()
# Calculate the execution time in seconds
execution_time = end_time - start_time
# Print the execution time
print("\n Total Execution time of Script:", execution_time/3600, "hours")
####################################################################################
