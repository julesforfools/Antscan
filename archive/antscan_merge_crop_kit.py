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
from skimage import io
import numpy as np
import scipy.ndimage # to perform processes after DNN segmentation
import time
import psutil # to monitor memory usage
import warnings
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
#sys.argv[4] is the location of the .h5 DNN

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
        "/2X/": (6.11, 6.11, 6.11),
        "/5x/": (2.44, 2.44, 2.44),
        "/10x/": (1.22, 1.22, 1.22),
        "/2x/": (6.11, 6.11, 6.11)
    }
    # Iterate through the keyphrases in path
    for path_fragment, spacing_values in keys.items():
        if path_fragment in image_path:
            # Set the voxel spacing for the image
            image.SetSpacing(spacing_values)
            return(image)

def load_scan(path):
    '''Load and return image in sitk format'''
    image = sitk.ReadImage(path)
    # Set the image spacing, origin, and direction if needed
    image.SetSpacing((1.0, 1.0, 1.0))  # Adjust spacing according to your image
    image.SetOrigin((0.0, 0.0, 0.0))  # Adjust origin according to your image
    image.SetDirection(np.eye(3).flatten())  # Adjust direction according to your image
    return(image)

def load_scan_sk(path):
    '''Load with skimage to avoid encountering unknown field tags and return image in sitk format for further processing'''
    image = io.imread(path)
    image = sitk.GetImageFromArray(image)
    print(image.GetSize())
    # Set the image spacing, origin, and direction if needed
    image.SetSpacing((1.0, 1.0, 1.0))  # Adjust spacing according to your image
    image.SetOrigin((0.0, 0.0, 0.0))  # Adjust origin according to your image
    image.SetDirection(np.eye(3).flatten())  # Adjust direction according to your image
    return(image)

def four_step_registration(fixed_image, moving_image, initial_z):

    '''
    This function registers two CT-stacks where the approximate z-step (in pixels) is known.
    If there is any kind of info like voxel-size attached to the files, it will probably fail.
    The function requires a lot of memory and was written with 3D-Tiff files in mind.
    The function returns nothing but an SimpleITK transform object

    this function can use downsampling to avoid large memory usage from casting to 32-bit

    this function was purely made for antscan data
    '''

    #############
    ### Setup ###
    #############
    ### Crop out parts of the images that are unnecessary for registration
    n_crop_remain = int((moving_image.GetSize()[2]-(initial_z))*1.05) # Choose the number of slices to remain after cropping: overlap plus five percent
    #print("Remaining Images after cropping:", n_crop_remain)
    moving_image = moving_image[:, :, -n_crop_remain:] # Crop sitk image by slicing, note the minus to indicate that we want the last images for the upper part
    #print(moving_image.GetSize())
    #print("Upper Cropped Image Origin:", moving_image.GetOrigin())
    fixed_image = fixed_image[:, :, :n_crop_remain] # Crop sitk image by slicing, note the minus to indicate that we want the last images for the upper part
    #print(fixed_image.GetSize())
    #print("Lower Cropped Image Origin:", fixed_image.GetOrigin())

    ### Cast input images to 32bit for registration
    # Convert the images to floating-point format
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    ### Setup of the initial transformation
    dimension = 3
    translation = sitk.TranslationTransform(dimension)
    translation.SetParameters((0.0, 0.0, float(initial_z)))
    #print("Initial Parameters: " + str(translation.GetOffset()))

    ##############################
    ### Four-Step registration ###
    ##############################
    ### Round 1: Exhaustive Registration on 16-times downsampling ###
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=255)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    # Select Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Create the exhaustive optimizer
    registration_method.SetOptimizerAsExhaustive(
        numberOfSteps=[4, 4, 4], stepLength=2 # Steps go in positive and negative direction
        )

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose the transform and Optimize in-place
    registration_method.SetInitialTransform(translation, inPlace=True) # Change here the types of transform being used for registration

    # Run the registration
    registration_method.Execute(fixed_image, moving_image)
    #print(f"Final metric value: {registration_method.GetMetricValue()}")
    #print(
    #    f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    #    )
    #print("Parameters after first step: " + str(tuple(elem for elem in translation.GetOffset())))


    ### Round 2: Exhaustive Registration on 16-times downsampling with a smaller spectrum ###
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=255)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    # Select Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Create the exhaustive optimizer
    registration_method.SetOptimizerAsExhaustive(
        numberOfSteps=[2, 2, 2], stepLength=1 # Steps go in positive and negative direction
        )

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[16])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose the transform and Optimize in-place
    registration_method.SetInitialTransform(translation, inPlace=True) # Change here the types of transform being used for registration

    # Run the registration
    registration_method.Execute(fixed_image, moving_image)
    #print(f"Final metric value: {registration_method.GetMetricValue()}")
    #print(
    #    f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    #    )
    #print("Parameters after second step: " + str(tuple(elem for elem in translation.GetOffset())))

    ### Round 3: Exhaustive Registration on 8-times downsampling finer ###
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=255)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    # Select Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Create the exhaustive optimizer
    registration_method.SetOptimizerAsExhaustive(
        numberOfSteps=[2, 2, 2], stepLength=0.25 # Steps go in positive and negative direction
        )

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose the transform and Optimize in-place
    registration_method.SetInitialTransform(translation, inPlace=True) # Change here the types of transform being used for registration

    # Run the registration
    registration_method.Execute(fixed_image, moving_image)
    #print(f"Final metric value: {registration_method.GetMetricValue()}")
    #print(
    #    f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    #    )
    #print("Parameters after third step: " + str(tuple(elem for elem in translation.GetOffset())))

    ### Round 4: Final step with Gradient Descent optimizer on 8,4,2 times downsampling ###
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=255)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1) #Only use that much percent of image pixels

    # Select Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        )
    #registration_method.SetOptimizerScalesFromPhysicalShift() # Don't use this setting!

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2]) # Can be read as times downsampling
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose the transform and Optimize in-place
    registration_method.SetInitialTransform(translation, inPlace=True) # Change here the types of transform being used for registration

    # Run the registration
    registration_method.Execute(fixed_image, moving_image)
    #print(f"Final metric value: {registration_method.GetMetricValue()}")
    #print(
    #    f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    #    )
    #print("Parameters after last step: " + str(tuple(elem for elem in translation.GetOffset())))

    ### Get values to apply to images
    translation.SetParameters((
        translation.GetOffset()[0],
        translation.GetOffset()[1],
        translation.GetOffset()[2],
    ))

    return(translation)


def merge_ct(lower_part, upper_part, transform):

    '''
    This is the antscan_merge_crop_kit version!!! It reads an sitk transform
    This function computes the resampling and merges two datasets into a new, stitched one.
    We need an SITK transform, it can not be a composite and probably also needs to be a translation-only.
    The function primarily returns the average merged image but additionally the larger canvas for automated cropping and masking
    '''

    dimension = 3

    # Get the size of the lower image
    size = lower_part.GetSize()

    ### Setup of the initial transformation
    #translation = sitk.TranslationTransform(dimension)
    #translation.SetParameters(transform)
    #print("Initial Parameters: " + str(translation.GetOffset()))

    # Compute the required size of the final image
    transformed_moving_size = [int((size[i] - 1) + abs(transform.GetOffset()[i])) + 1 for i in range(3)]
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
        transform.GetInverse(), # Apply inverse transform as the origin is the upper corner
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

    del upper_part
    del lower_part
    # cast back to sitk
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
    Changed to read in numpy array image
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

    ############################################################### Start a timer
    dil_start_time = time.time()
    ################################################################################
    # Perform dilation on the segmentation array
    mask = scipy.ndimage.binary_dilation(mask, structure = structuring_element, iterations=iterations)
    ################################################################ End the timer
    dil_end_time = time.time()
    # Calculate the execution time in seconds
    dil_execution_time = dil_end_time - dil_start_time
    # Print the execution time
    print("\n Initial Dilation time:", dil_execution_time, "seconds")
    ####################################################################################

    ############################################################### Start a timer
    dil_start_time = time.time()
    ################################################################################
    # Largest islands segmentation with scipy
    mask, num_features = scipy.ndimage.label(mask, output = np.int16)
    component_sizes = np.bincount(mask.ravel())
    largest_component_index = np.argmax(component_sizes[1:]) + 1
    mask = (mask == largest_component_index).astype(np.uint8)
    ################################################################ End the timer
    dil_end_time = time.time()
    # Calculate the execution time in seconds
    dil_execution_time = dil_end_time - dil_start_time
    # Print the execution time
    print("\n Largest Island Selection time:", dil_execution_time, "seconds")
    ####################################################################################

    ############################################################### Start a timer
    dil_start_time = time.time()
    ################################################################################
    ### Create a buffer around the segmentation
    # Define the size of the buffer
    size = int(dil_it*3)
    # Perform dilation on the segmentation array
    #mask = scipy.ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)

    ### Mask & Crop
    # Find the indices where the mask is non-zero
    nonzero_indices = np.nonzero(mask)
    # Determine the minimum and maximum indices along each dimension
    min_z, min_y, min_x = np.min(nonzero_indices, axis=1)
    max_z, max_y, max_x = np.max(nonzero_indices, axis=1)
    ### Apply
    #print("Image size before:", image.GetSize())
    ############################
    # Apply buffer
    min_z, min_y, min_x = (min_z - size), (min_y - size), (min_x - size)
    max_z, max_y, max_x = (max_z + size), (max_y + size), (max_x + size)

    # Check if buffer exceeds image boundaries and issue a warning
    if min_z < 0:
        warnings.warn("Warning: Buffer exceeds minimum index along z-axis. Setting min_z to 0.")
        min_z = 0
    if min_y < 0:
        warnings.warn("Warning: Buffer exceeds minimum index along y-axis. Setting min_y to 0.")
        min_y = 0
    if min_x < 0:
        warnings.warn("Warning: Buffer exceeds minimum index along x-axis. Setting min_x to 0.")
        min_x = 0
    if max_z >= image.shape[0]:
        warnings.warn(f"Warning: Buffer exceeds maximum index along z-axis. Setting max_z to {image.shape[0] - 1}.")
        max_z = image.shape[0] - 1
    if max_y >= image.shape[1]:
        warnings.warn(f"Warning: Buffer exceeds maximum index along y-axis. Setting max_y to {image.shape[1] - 1}.")
        max_y = image.shape[1] - 1
    if max_x >= image.shape[2]:
        warnings.warn(f"Warning: Buffer exceeds maximum index along x-axis. Setting max_x to {image.shape[2] - 1}.")
        max_x = image.shape[2] - 1

    # Mask
    ### Errors were reported ###
    #image = image*mask

    ############################
    #print("Image size before:", image.GetSize())
    # Crop the image to the bounding box of the segmentation
    image = image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    ################################################################ End the timer
    dil_end_time = time.time()
    # Calculate the execution time in seconds
    dil_execution_time = dil_end_time - dil_start_time
    # Print the execution time
    print("\n Final Dilation time:", dil_execution_time, "seconds")
    ####################################################################################

    # Convert back to sitk
    image = sitk.GetImageFromArray(image)
    print("Image size after:", image.GetSize())

    return(image, [min_x, max_x, min_y, max_y, min_z, max_z])



############################################################################################
############################### Script #####################################################
############################################################################################
parent_folder = sys.argv[1]

folders = absolute_folder_paths(parent_folder) # get folders/specimens
folders = sorted(folders) # sort folders alphabetically!
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
print(specimens)
start_spec = 0 # for subsetting if time is limited, first number must be the same as ind
end_spec = int(len(specimens)-1)
specimens = specimens[start_spec:]# for subsetting if time is limited, first number must be the same as ind
# write out specimens to file
with open(sys.argv[3]+"specimens.txt", 'w') as fp:
    for item in specimens:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

############################################################### Start a timer
start_time = time.time()
################################################################################

#############################################################################################
#################### Register, merge, crop all specimens in folder ##########################
#############################################################################################
for specimen in specimens:

    ############################################################### Start a timer
    spec_start_time = time.time()
    ################################################################################

    spec_name = os.path.basename(specimen[0]).split('_')[0]
    print("\n\n Next specimen:", spec_name)
    main_image_path = specimen[-1] # The highest z-Stage is the 0,0,0 origin
    main_image = load_scan_sk([entry.path for entry in os.scandir(main_image_path) if entry.is_file()][-1]) # In antscan, the blend file comes after the abs file!
    z_shift = 0 # initialize z-shift

    #############################################
    transform_data = [] # Create empty list to append to specimen after registration
    ########## Register & Merge Images ##########
    while(len(specimen) > 1 and type(specimen) == list): # As long as there's images left to merge
        specimen.pop() # Remove the last element with pop method lmao
        lower_image_path = specimen[-1]
        lower_image = load_scan_sk([entry.path for entry in os.scandir(lower_image_path) if entry.is_file()][-1])

        z_shift = z_shift + int(sys.argv[2]) # z_shift depends on magnification: 5x: 1650, 2x: 820; multiply these values for further z-stages
        # Registration
        print("Main Image Size: ", main_image.GetSize())
        print("Lower Image Size: ", lower_image.GetSize())
        final_transform = four_step_registration(lower_image, main_image, z_shift)
        print("Transform values: ", final_transform.GetOffset())
        # Merge
        main_image = merge_ct(lower_image, main_image, final_transform) # Keep item as main_image for while loop
        del lower_image #deallocate object to save memory
        # Update z_shift
        z_shift = final_transform.GetOffset()[2]
        transform_data.append(final_transform.GetOffset())
    #############################################
    specimen.append(transform_data) # append transform results to specimen

    ###########################################################################
    #################### Mask the images with segmentation ####################
    sitk.WriteImage(main_image, sys.argv[3]+"intermediary_result.nii") # Temporarily store results
    del main_image
    #sitk.WriteImage(main_image, sys.argv[3]+spec_name+"_intermediary_result.nii") # Temporarily store results
    ##################################
    ### Biomedisa DNN Segmentation ###

    # load data
    main_image, img_header, img_ext = load_data(sys.argv[3]+"intermediary_result.nii",
        return_extension=True)
    # deep learning
    results = deep_learning(main_image, predict=True, img_header=img_header,
        path_to_model=sys.argv[4], img_extension=img_ext, batch_size = 6)
    #mask = refinement(img, results['regular'])
    mask = results['regular']
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
    del mask
    specimen.append(indices) #append cropping indices to specimen
    #print(image_clean.GetPixelIDTypeAsString())

    ###########################################################################

    ####################################
    ########## Set Voxel Size ##########
    # Use the image path to feel out the voxel size and adjust using the helper function
    #main_image = set_voxel_spacing(main_image_path, main_image)
    print(main_image.GetSpacing())
    ####################################

    OUTPUT = sys.argv[3]+spec_name+".tif"
    sitk.WriteImage(main_image, OUTPUT)
    del main_image

    ### Write Registration results to file ###
    # open file in append mode
    with open(sys.argv[3]+"specimens_processed_"+str(start_spec)+"-"+str(end_spec)+".txt", 'a') as fp:
        fp.write("%s\n" % specimen)
        print('Writing Results to file for '+spec_name+' Done')

    ################################################################ End the timer
    spec_end_time = time.time()
    # Calculate the execution time in seconds
    spec_execution_time = spec_end_time - spec_start_time
    # Print the execution time
    print("\n Total time for specimen: ", spec_name, " ", spec_execution_time/60, "minutes")
    ####################################################################################

################################################################ End the timer
end_time = time.time()
# Calculate the execution time in seconds
execution_time = end_time - start_time
# Print the execution time
print("\n Total Execution time of Script:", execution_time/3600, "hours")
####################################################################################
