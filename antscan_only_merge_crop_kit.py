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
import SimpleITK as sitk
import numpy as np
import scipy.ndimage # to perform processes after DNN segmentation
#import time
import psutil # to monitor memory usage
import ast

### Setup Biomedisa (version 23.08.1 or higher)
# change this line to your biomedisa directory
#path_to_biomedisa = '/apps/unit/EconomoU/biomedisa_dnn/bin/git/biomedisa'
path_to_biomedisa = sys.argv[5]
# load libraries
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data, save_data
from demo.biomedisa_deeplearning import deep_learning
from demo.keras_helper import get_image_dimensions, get_physical_size
from biomedisa_features.active_contour import refinement

print(sys.argv[0:])
#print(len(sys.argv))
#sys.argv is a list in Python, which contains the command-line arguments passed to the script.
#sys.argv[0] is the script, sys.argv[1] is the source file directory, sys.argv[2] is the z-Shift, sys.argv[3] is the destination file directory,
#sys.argv[4] is the location of the .h5 DNN, sys.argv[5] is the path to biomedisa

############################################################################################
############################### Functions ##################################################
############################################################################################

def absolute_folder_paths(directory):
    ''' Return Paths of all folders (scans) in parent folder'''
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path)]

def set_voxel_spacing(image_path, image):
    '''Helper Function to adjust image spacing in antscan data so users don't have to'''
    '''Note for Amira/Avizo Users: To apply the new spacing, 'Resample Transformed Image' with settings to 'Extended' and 'Preserve Dimensions' '''
    # Define the mapping of paths to voxel spacings
    keys = {
        "/5X/": (2.44, 2.44, 2.44),
        "/10X/": (1.22, 1.22, 1.22),
        "/2X/": (6.11, 6.11, 6.11)
    }
    # Iterate through the keyphrases in path
    for path_fragment, spacing_values in keys.items():
        if path_fragment in image_path:
            # Set the voxel spacing for the image
            image.SetSpacing(spacing_values)
            return(image)

def ReadRegistrationResults(filename):
    '''Helper Function to load a previous version of 'specimens' with registration results printed out to file'''
    result_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Using ast.literal_eval to safely evaluate the string representation of arrays
            result = ast.literal_eval(line.strip())
            result_list.append(result)
    return result_list

def load_scan(path):
    '''Load and return image in sitk format'''
    image = sitk.ReadImage(path)
    # Set the image spacing, origin, and direction if needed
    image.SetSpacing((1.0, 1.0, 1.0))  # Adjust spacing according to your image
    image.SetOrigin((0.0, 0.0, 0.0))  # Adjust origin according to your image
    image.SetDirection(np.eye(3).flatten())  # Adjust direction according to your image
    return(image)

def merge_ct(lower_part, upper_part, transform):

    '''
    This function computes the resampling and merges two datasets into a new, stitched one.
    We need an SITK transform, it can not be a composite and probably also needs to be a translation-only.
    The function primarily returns the average merged image but additionally the larger canvas for automated cropping and masking
    '''

    dimension = 3

    # Get the size of the lower image
    size = lower_part.GetSize()

    ### Setup of the initial transformation
    translation = sitk.TranslationTransform(dimension)
    translation.SetParameters(transform)
    #print("Initial Parameters: " + str(translation.GetOffset()))

    # Compute the required size of the final image
    transformed_moving_size = [int((size[i] - 1) + abs(translation.GetOffset()[i])) + 1 for i in range(3)]
    #print("Target Size", transformed_moving_size)
    transformed_moving_size.reverse() # Reverse for numpy array order of dimensions
    canvas = np.zeros(transformed_moving_size, dtype=np.uint8)
    canvas = sitk.GetImageFromArray(canvas)
    #print("Numpy Array Shape",canvas_np.shape) # removed for memory
    #print("ITK Canvas Size", canvas.GetSize())

    n_overlap = upper_part.GetSize()[2]+lower_part.GetSize()[2]-canvas.GetSize()[2] #Calculate overlapping region
    end_upper_part = (upper_part.GetSize()[2])-1 #Save the number of slices in the upper image as an index of the beginning of the overlap
    #print("\n The overlapping slices total ", n_overlap, "and the index for the last overlapping slice is", end_upper_part)

    # Resample the images statically onto the new merged size
    # The upper part first
    upper_part = sitk.Resample(
        upper_part,
        canvas,
        sitk.Transform(dimension, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        upper_part.GetPixelID(),
    )
    # Next the lower part
    lower_part = sitk.Resample(
        lower_part,
        canvas,
        translation.GetInverse(), # Apply inverse transform as the origin is the upper corner
        sitk.sitkLinear,
        0.0,
        lower_part.GetPixelID(),
    )

    upper_part = sitk.GetArrayFromImage(upper_part)
    lower_part = sitk.GetArrayFromImage(lower_part)

    merged_image = upper_part+lower_part

    # Compute weighted average as a gradient within the overlapping region
    for slice in range(n_overlap):
        merged_image[end_upper_part - slice] = slice/(n_overlap-1) * upper_part[end_upper_part - slice] + ((n_overlap-1)-slice)/(n_overlap-1) * lower_part[end_upper_part - slice]

    merged_image = sitk.GetImageFromArray(merged_image)

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("Peak Memory usage in Merging step:", memory_usage, "GB")
    ###

    #print("New Merged Image Size:", merged_image.GetSize())
    #print("Pixel Type:", merged_image.GetPixelIDTypeAsString())
    print("Merging Images Done!")

    return(merged_image)

def mask2crop2(image, mask, dil_it=10):

    '''
    This is the scipy version of the code
    This function computes masking and cropping of the averaged image
    This is a new version of the function to simply take in a file and a corresponding segmentation.
    Included here is a largest island selection to filter some noise in the segmentation.
    To dilate the result more, we set the argument of dilation iterations.
    !!! Due to high number of small errors, masking removed !!!
    '''

    #mask = sitk.GetArrayFromImage(mask) #Deprecated as biomedisa stores mask as Numpy Array
    ### Initial dilation to better connect components
    # Create a structuring element for dilation
    structuring_element = scipy.ndimage.generate_binary_structure(3, 1)
    # Define the number of iterations for dilation
    iterations = int(dil_it)

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("Memory usage before dilating segmentation:", memory_usage, "GB")
    ###

    # Perform dilation on the segmentation array
    mask = scipy.ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)

    # Largest islands segmentation with scipy
    mask, num_features = scipy.ndimage.label(mask)
    component_sizes = np.bincount(mask.ravel())
    largest_component_index = np.argmax(component_sizes[1:]) + 1
    mask = (mask == largest_component_index).astype(np.uint8)

    ### Dilate again to create a buffer around the segmentation
    # Define the number of iterations for dilation
    iterations = int(dil_it*3)
    # Perform dilation on the segmentation array
    mask = scipy.ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)

    ### Mask & Crop
    # Find the indices where the mask is non-zero
    nonzero_indices = np.nonzero(mask)
    # Determine the minimum and maximum indices along each dimension
    min_z, min_y, min_x = np.min(nonzero_indices, axis=1)
    max_z, max_y, max_x = np.max(nonzero_indices, axis=1)
    ### Apply
    #print("Image size before:", image.GetSize())
    ############################
    # Mask first to be relatively sure that the bright vial-air-boundary is not in the final result
    ### Errors were reported ###
    image = (sitk.GetArrayFromImage(image))#*mask
    ############################
    # Crop the image to the bounding box of the segmentation
    image = image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    # Convert back to sitk
    image = sitk.GetImageFromArray(image)
    #print("Image size after:", image.GetSize())

    return(image, [min_x, max_x, min_y, max_y, min_z, max_z])



############################################################################################
############################### Script #####################################################
############################################################################################
parent_folder = sys.argv[1]

folders = absolute_folder_paths(parent_folder) # get folders/specimens
folders = sorted(folders)
print("Total number of folders:", len(folders))

specimens = []
for path in folders:
    subfolder = os.path.basename(path)  # Get the ultimate subfolder
    left_part = subfolder.split('_')[0]  # Extract the left part before the underscore

    # Check if there is an existing list for the left part
    found = False
    for sublist in specimens:
        if os.path.basename(sublist[0]).split('_')[0] == left_part:
            sublist.append(path)
            found = True
            break

    # If no existing list is found, create a new one
    if not found:
        specimens.append([path])

print("Total number of specimens:", len(specimens))
# specimens = specimens[100:300]# for subsetting if time is limited, first number must be the same as ind
# write out specimens to file
with open(sys.argv[3]+"specimens.txt", 'w') as fp:
    for item in specimens:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

################################# Load Registration Results #################################
if sys.argv[6]:
    registration_results = ReadRegistrationResults(sys.argv[6])
ind = 0 # Only works if order is the same
#############################################################################################
#############################################################################################
############### Don't Register, but merge, crop all specimens in folder ###############
for specimen in specimens:

    spec_name = os.path.basename(specimen[0]).split('_')[0]
    print("\n\n Next specimen:", spec_name)
    main_image_path = specimen[-1] #The highest z-Stage is the 0,0,0 origin
    main_image = load_scan([entry.path for entry in os.scandir(main_image_path) if entry.is_file()][-1]) # In antscan, the blend file comes after the abs file!
    z_shift = 0 # initialize z-shift

    #############################################
    print("Current Registration results for: ", os.path.basename(registration_results[ind][0]).split('_')[0])
    # If order doesn't match, can use spec_name to identify position of current specimen in registration_results
    ###########################################################################
    #################### Register & Merge Images ##############################
    while(len(specimen) > 1 and type(specimen) == list): # As long as there's images left to merge
        final_transform = registration_results[ind][1][0]
        print(final_transform)
        specimen.pop() # Remove the last element with pop method lmao
        registration_results[ind][1].pop(0) # Remove the first element of the transform list with pop method lmao
        lower_image_path = specimen[-1]
        lower_image = load_scan([entry.path for entry in os.scandir(lower_image_path) if entry.is_file()][-1])
        # Merge
        main_image = merge_ct(lower_image, main_image, final_transform) # Keep item as main_image for while loop
        del lower_image #deallocate object to save memory
    #############################################
    #specimen.append(transform_data) # append transform results to specimen

    ###########################################################################
    #################### Mask the images with segmentation ####################
    sitk.WriteImage(main_image, sys.argv[3]+"intermediary_result.nii") # Temporarily store results
    #sitk.WriteImage(main_image, sys.argv[3]+spec_name+"_intermediary_result.nii") # Temporarily store results
    ##################################
    ### Biomedisa DNN Segmentation ###

    # load data
    img, img_header, img_ext = load_data(sys.argv[3]+"intermediary_result.nii",
        return_extension=True)
    # deep learning
    results = deep_learning(img, predict=True, img_header=img_header,
        #path_to_model='/home/j/jkatzke/antscan.h5', img_extension=img_ext)
        path_to_model=sys.argv[4], img_extension=img_ext)
    mask = refinement(img, results['regular'])
    del results #deallocate object to save memory
    #print("results_refined:", type(mask))
    ##################################

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("Memory usage before Masking step:", memory_usage, "GB")
    ###

    ### Apply mask ###
    main_image, indices = mask2crop2(main_image, mask)
    specimen.append(indices) #append cropping indices to specimen
    #print(image_clean.GetPixelIDTypeAsString())
    ###########################################################################

    ####################################
    ########## Set Voxel Size ##########
    # Use the image path to feel out the voxel size and adjust using the helper function
    main_image = set_voxel_spacing(main_image_path, main_image)
    print(main_image.GetSpacing())
    ####################################

    OUTPUT = sys.argv[3]+spec_name+".nii"
    sitk.WriteImage(main_image, OUTPUT)
    ind = ind+1
