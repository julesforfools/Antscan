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

def mask2mask(mask_path):

    '''
    This function reads a binary image and transforms its values to zeros and ones.
    If the mask is already zeros and ones, then this is superfluous.
    It is meant to be used with a tiff file.
    It assumes the background has more pixels in the image than the foreground.
    '''

    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    # Investigate the mask
    mask = sitk.GetArrayFromImage(mask)
    mask_flat = mask.flatten()

    # Get the unique values and their counts
    unique_values, counts = np.unique(mask_flat, return_counts=True)

    # Sort the unique values and counts in descending order based on counts
    sorted_indices = np.argsort(-counts)
    unique_values_sorted = unique_values[sorted_indices]

     # Check if there are exactly two unique values
    if len(unique_values_sorted) == 2:
        mask[mask == unique_values_sorted[0]] = 0 # Background pixels are set to zero
        mask[mask == unique_values_sorted[1]] = 1 # Foreground pixels are set to one
        mask = sitk.GetImageFromArray(mask)
        return(mask)
    else:
        raise ValueError("Expected exactly two unique values, but found otherwise. The input does not seem to be a mask")

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

def mask2crop(average_image, canvas, transform, fixed_image_mask, moving_image_mask, dil_it=5):

    '''
    This function computes masking and cropping of the averaged image
    We need the masks from mask2mask, the average image and the canvas from merge_ct, and the transform from four_step_registration
    To dilate the result in case hairs or something were not included in the original mask, we set and additional argument of dilation iterations
    '''

    ### Setup Masks
    dimension = 3
    # Use the original masks for pixels that are overlapping
    moving_image_mask = sitk.Resample(
        moving_image_mask,
        canvas,
        sitk.Transform(dimension, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor, # Interpolation must be absolute
        0.0,
        moving_image_mask.GetPixelID(),
    )
    moving_image_mask = sitk.GetArrayFromImage(moving_image_mask)

    fixed_image_mask = sitk.Resample(
        fixed_image_mask,
        canvas,
        transform.GetInverse(), # Apply inverse transform as the origin is the upper corner
        sitk.sitkNearestNeighbor, # Interpolation must be absolute
        0.0,
        fixed_image_mask.GetPixelID(),
    )
    fixed_image_mask = sitk.GetArrayFromImage(fixed_image_mask)

    print("Upper part Mask Size:", moving_image_mask.shape, "Lower Part Mask Size:", fixed_image_mask.shape)

    # Threshold the two overlapping masks to only get signal (1) and background (0)
    mask_full = np.where(moving_image_mask+fixed_image_mask >= 1, 1, 0)

    # Create a structuring element for dilation
    structuring_element = scipy.ndimage.generate_binary_structure(3, 1)
    # Define the number of iterations for dilation
    iterations = dil_it
    # Perform dilation on the segmentation array
    dilated_mask = scipy.ndimage.binary_dilation(mask_full, structure=structuring_element, iterations=iterations)

    ### Crop
    # Find the indices where the mask is non-zero
    nonzero_indices = np.nonzero(dilated_mask)
    # Determine the minimum and maximum indices along each dimension
    min_z, min_y, min_x = np.min(nonzero_indices, axis=1)
    max_z, max_y, max_x = np.max(nonzero_indices, axis=1)

    ### Apply
    # Mask first to be relatively sure that the bright vial-air-boundary is not in the final result
    average_image = (sitk.GetArrayFromImage(average_image))*dilated_mask
    # Crop the image to the bounding box of the segmentation
    average_image = average_image[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    print(average_image.shape)

    # Convert back to sitk
    average_image = sitk.GetImageFromArray(average_image)
    print(average_image.GetSize())

    return(average_image)



############################################################################################
############################### Script #####################################################
############################################################################################
# Start a timer
start_time = time.time()

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
    
    spec_name = os.path.basename(specimen[0]).split('_')[0]
    print("Next specimen:", spec_name)
    upper_image_path = specimen[-1]
    upper_image = load_scan([entry.path for entry in os.scandir(upper_image_path) if entry.is_file()][-1])
    z_shift = 0 # initialize z-shift
    
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
  
    #######################################################
    ########## Mask the images with segmentation ##########
    #image_clean = mask2crop(upper_image, canvas, final_transform, mask_00, mask_01)
    #print(average_image_clean.GetPixelIDTypeAsString())
    #######################################################
        
    OUTPUT = sys.argv[3]+spec_name+".nii"
        
    sitk.WriteImage(upper_image, OUTPUT)
    #sitk.WriteImage(image_clean, OUTPUT)



# End the timer
end_time = time.time()
# Calculate the execution time in seconds
execution_time = end_time - start_time
# Print the execution time
print("\nExecution time:", execution_time, "seconds")
