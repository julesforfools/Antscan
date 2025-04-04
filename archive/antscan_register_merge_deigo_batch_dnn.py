# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:19:07 2023

@author: Julian Katzke
"""

############################################################################################
############################### Setup ######################################################
############################################################################################

import os
import sys
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import time
import psutil

print(sys.argv[0:])
#print(len(sys.argv))
#sys.argv is a list in Python, which contains the command-line arguments passed to the script.
#sys.argv[0] is the directory, sys.argv[1] is the script, etc..


############################################################################################
############################### Functions ##################################################
############################################################################################

def absolute_folder_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path)]

def load_scan(path):
    '''Load and return image in sitk format'''
    image = sitk.ReadImage(path)
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

    ### Crop out parts of the images that are unnecessary for registration
    n_crop_remain = int((moving_image.GetSize()[2]-(initial_z))*1.10) # Choose the number of slices to remain after cropping: overlap plus ten percent
    print("Remaining Images after cropping:", n_crop_remain)
    moving_image = moving_image[:, :, -n_crop_remain:] # Crop sitk image by slicing, note the minus to indicate that we want the last images for the upper part
    print(moving_image.GetSize())
    print("Upper Cropped Image Origin:", moving_image.GetOrigin())
    fixed_image = fixed_image[:, :, :n_crop_remain] # Crop sitk image by slicing, note the minus to indicate that we want the last images for the upper part
    print(fixed_image.GetSize())
    print("Lower Cropped Image Origin:", fixed_image.GetOrigin())

    ### Cast input images to 32bit for registration
    # Convert the images to floating-point format
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    ### Setup of the initial transformation
    dimension = 3
    translation = sitk.TranslationTransform(dimension)
    translation.SetParameters((0.0, 0.0, float(initial_z)))
    print("Initial Parameters: " + str(translation.GetOffset()))

    ###############################
    ### Three-Step registration ###
    ###############################
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
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
    print("Parameters after first step: " + str(tuple(elem for elem in translation.GetOffset())))


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
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
    print("Parameters after second step: " + str(tuple(elem for elem in translation.GetOffset())))

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
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
    print("Parameters after third step: " + str(tuple(elem for elem in translation.GetOffset())))

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
    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
    print("Parameters after last step: " + str(tuple(elem for elem in translation.GetOffset())))

    ### Scale Translation Up to apply to images
    translation.SetParameters((
        translation.GetOffset()[0],
        translation.GetOffset()[1],
        translation.GetOffset()[2],
    ))

    return(translation)


def merge_ct(lower_part, upper_part, transform):

    '''
    This function computes the resampling and merges two datasets into a new, stitched one.
    We need an SITK transform, it can not be a composite and probably also needs to be a translation-only.
    The function primarily returns the average merged image but additionally the larger canvas for automated cropping and masking
    '''

    dimension = 3

    # Get the size of the fixed image
    size = lower_part.GetSize()

    # Compute the required size of the resampled image
    transformed_moving_size = [int((size[i] - 1) + abs(transform.GetOffset()[i])) + 1 for i in range(3)]
    print("Target Size", transformed_moving_size)
    transformed_moving_size.reverse() # Reverse for numpy array order of dimensions
    canvas = np.zeros(transformed_moving_size, dtype=np.uint8)
    canvas = sitk.GetImageFromArray(canvas)
    #print("Numpy Array Shape",canvas_np.shape) # removed for memory
    print("ITK Canvas Size", canvas.GetSize())

    n_overlap = upper_part.GetSize()[2]+lower_part.GetSize()[2]-canvas.GetSize()[2] #Calculate overlapping region
    end_upper_part = (upper_part.GetSize()[2])-1 #Save the number of slices in the upper image as an index of the beginning of the overlap
    print("\n The overlapping slices total ", n_overlap, "and the index for the last overlapping slice is", end_upper_part)

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

    merged_image = sitk.GetImageFromArray(merged_image)

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("Peak Memory usage in Merging step:", memory_usage, "GB")
    ###

    print("New Merged Image Size:", merged_image.GetSize())
    print("Pixel Type:", merged_image.GetPixelIDTypeAsString())
    print("Merging Images Done!\n")

    return(merged_image)

def mask2crop2(image, mask, dil_it=10):

    '''
    This is the scipy version of the code
    This function computes masking and cropping of the averaged image
    This is a new version of the function to simply take in a file and a corresponding segmentation.
    Included here is a largest island selection to filter some noise in the segmentation.
    To dilate the result in case hairs or something were not included in the original mask, we set and additional argument of dilation iterations
    '''

    ### Setup Masks
    dimension = 3

    ### Perform largest island selection from Ziv Yaniv directly
    mask = sitk.GetArrayFromImage(mask)
    # Dilate first to capture small details like legs and feet
    # Create a structuring element for dilation
    structuring_element = scipy.ndimage.generate_binary_structure(3, 1)
    # Define the number of iterations for dilation
    iterations = int(dil_it)
    # Perform dilation on the segmentation array
    mask = scipy.ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)
    # Largest islands with scipy
    labeled_array, num_features = scipy.ndimage.label(mask)
    component_sizes = np.bincount(labeled_array.ravel())
    largest_component_index = np.argmax(component_sizes[1:]) + 1
    mask = (labeled_array == largest_component_index).astype(int)
    # Dilate again
    # Define the number of iterations for dilation
    iterations = int(dil_it*2)
    # Perform dilation on the segmentation array
    mask = scipy.ndimage.binary_dilation(mask, structure=structuring_element, iterations=iterations)

    ### Mask & Crop
    # Find the indices where the mask is non-zero
    nonzero_indices = np.nonzero(mask)
    # Determine the minimum and maximum indices along each dimension
    min_z, min_y, min_x = np.min(nonzero_indices, axis=1)
    max_z, max_y, max_x = np.max(nonzero_indices, axis=1)
    ### Apply
    print("Image size before:", image.GetSize())
    # Mask first to be relatively sure that the bright vial-air-boundary is not in the final result
    image = (sitk.GetArrayFromImage(image))*mask
    # Crop the image to the bounding box of the segmentation
    image = image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    # Convert back to sitk
    image = sitk.GetImageFromArray(image)
    print("Image size after:", image.GetSize())

    return(image)



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

# Register, merge, crop all specimens in folder
for specimen in specimens:
    # Start a timer
    start_time = time.time()

    spec_name = os.path.basename(specimen[0]).split('_')[0]
    print("Next specimen:", spec_name)
    upper_image_path = specimen[-1] #In antscan, the blend file comes after the abs file!
    upper_image = load_scan([entry.path for entry in os.scandir(upper_image_path) if entry.is_file()][-1])
    z_shift = 0 # initialize z-shift

    #############################################
    ########## Register & Merge Images ##########
    while(len(specimen) > 1 and type(specimen) == list): # As long as there's images left to merge
        specimen.pop() # Remove the last element with pop method lmao
        lower_image_path = specimen[-1]
        lower_image = load_scan([entry.path for entry in os.scandir(lower_image_path) if entry.is_file()][-1])

        z_shift = z_shift + int(sys.argv[2]) # z_shift depends on magnification: 5x: 1650, 2x: 820; multiply these values for further z-stages
        # Registration
        final_transform = four_step_registration(lower_image, upper_image, z_shift)
        # Merge
        upper_image = merge_ct(lower_image, upper_image, final_transform) # Keep item as upper_image for while loop
        # Update z_shift
        z_shift = final_transform.GetOffset()[2]
        print(final_transform.GetOffset())
    #############################################

    #######################################################
    ########## Mask the images with segmentation ##########
    sitk.WriteImage(upper_image, sys.argv[3]+"intermediary_result.nii") # Temporarily store results

    os.system('python3 /apps/unit/EconomoU/biomedisa_dnn/bin/git/biomedisa/demo/biomedisa_deeplearning.py -p {} antscan.h5'.format(sys.argv[3]+"intermediary_result.nii"))
    mask = sitk.ReadImage(sys.argv[3]+"final.intermediary_result.tif", sitk.sitkUInt8)
    print(mask.GetSize())
    image_clean = mask2crop2(upper_image, mask)
    print(image_clean.GetPixelIDTypeAsString())
    #######################################################

    OUTPUT = sys.argv[3]+spec_name+".nii"
    sitk.WriteImage(image_clean, OUTPUT)

    # End the timer
    end_time = time.time()
    # Calculate the execution time in seconds
    execution_time = end_time - start_time
    # Print the execution time
    print("\nExecution time:", execution_time, "seconds")
