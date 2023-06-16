# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:19:07 2023

@author: Julian Katzke
"""

############################################################################################
############################### Setup ######################################################
############################################################################################

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

fixed_image_path = sys.argv[1]
moving_image_path = sys.argv[2]
#fixed_mask_path = "G:/3d_workdir/ant_sf/5_antscan_stitch_tests/14-32/43-8_z_00_resample.tif.mask.tif"
#moving_mask_path = "G:/3d_workdir/ant_sf/5_antscan_stitch_tests/14-32/43-8_z_01_resample.tif.mask.tif"

OUTPUT = sys.argv[3]
z_shift = int(sys.argv[4]) # z_shift depends on magnification: 5x: 1650, 2x: 820; multiply these values for further z-stages

############################################################################################
############################### Functions ##################################################
############################################################################################

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

def downsample(image, fac=2):

    '''
    This function converts an image to a smaller version by resizing
    the canvas of the image and using linear interpolation in SimpleITK
    '''
    euler3d = sitk.Euler3DTransform()

    output_size = tuple(int((1/fac) * elem) for elem in image.GetSize()) # Use a rounded result for each dimension of the image
    output_spacing = tuple(fac * elem for elem in image.GetSpacing()) # Keep isometric voxel size

    image_resampled = sitk.Resample(
        image,
        output_size,
        euler3d,
        sitk.sitkLinear,
        image.GetOrigin(),
        output_spacing,
        image.GetDirection()
    )

    return(image_resampled)

def four_step_registration(fixed_image, moving_image, initial_z, factor=4):

    '''
    This function registers two CT-stacks where the approximate z-step (in pixels) is known.
    If there is any kind of info like voxel-size attached to the files, it will probably fail.
    The function requires a lot of memory and was written with 3D-Tiff files in mind.
    The function returns nothing but an SimpleITK transform object

    this function can use downsampling to avoid large memory usage from casting to 32-bit

    this function was purely made for antscan data
    '''

    ### Downsample image to not crash the memory of the computer
    if not factor==1:
        fixed_image = downsample(fixed_image, factor)
        moving_image = downsample(moving_image, factor)

    ### Cast input images to 32bit for registration
    # Convert the images to floating-point format
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    ### Setup of the initial transformation
    dimension = 3
    translation = sitk.TranslationTransform(dimension)
    translation.SetParameters((0.0, 0.0, float(initial_z/factor))) # Consider downsampling factor here
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
    print("Parameters after first step: " + str(tuple(factor * elem for elem in translation.GetOffset())))


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
    print("Parameters after second step: " + str(tuple(factor * elem for elem in translation.GetOffset())))

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
    print("Parameters after third step: " + str(tuple(factor * elem for elem in translation.GetOffset())))

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
    print("Parameters after last step: " + str(tuple(factor * elem for elem in translation.GetOffset())))

    ### Scale Translation Up to apply to images
    translation.SetParameters((
        translation.GetOffset()[0]*factor,
        translation.GetOffset()[1]*factor,
        translation.GetOffset()[2]*factor,
    ))

    return(translation)


def merge_ct(fixed_image, moving_image, transform):

    '''
    This function computes the resampling and merges two datasets into a new, stitched one.
    We need an SITK transform, it can not be a composite and probably also needs to be a translation-only.
    The function primarily returns the average merged image but additionally the larger canvas for automated cropping and masking
    '''

    dimension = 3

    # Get the size and spacing of the fixed image
    size = fixed_image.GetSize()
    spacing = fixed_image.GetSpacing()

    # Compute the required size of the resampled image
    transformed_moving_size = [int((size[i] - 1) * spacing[i] + abs(transform.GetOffset()[i])) + 1 for i in range(3)]
    print("Target Size", transformed_moving_size)
    transformed_moving_size.reverse() # Reverse for numpy array order of dimensions
    canvas = np.zeros(transformed_moving_size, dtype=np.uint8)
    canvas = sitk.GetImageFromArray(canvas)
    #print("Numpy Array Shape",canvas_np.shape) # removed for memory
    print("ITK Canvas Size", canvas.GetSize())

    # Delete the 0-15 slices of the upper image. The overlap is larger than that and they look bad usually!
    fixed_image = sitk.GetArrayFromImage(fixed_image) # Convert the image to a numpy array
    fixed_image[:10, :, :] = 0 # Set the first ten slices to black (zero)
    fixed_image = sitk.GetImageFromArray(fixed_image) # Convert the modified numpy array back to SimpleITK image

    # Resample the images statically onto the new merged size
    # The upper part first
    moving_image = sitk.Resample(
        moving_image,
        canvas,
        sitk.Transform(dimension, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )
    # Next the lower part
    fixed_image = sitk.Resample(
        fixed_image,
        canvas,
        transform.GetInverse(), # Apply inverse transform as the origin is the upper corner
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    ### Using the Overlapping areas to create an averaged image
    # Create masks for pixels that are overlapping
    mask1 = np.where(sitk.GetArrayFromImage(fixed_image) > 0, 1, 0).astype(np.uint8) # non-absolute background are 1
    mask2 = np.where(sitk.GetArrayFromImage(moving_image) > 0, 1, 0).astype(np.uint8) # non-absolute background are 1

    overlap = np.where(mask1+mask2 >= 2, 2, 1).astype(np.uint8) # to avoid dividing by 0, we set the background and non overlap to 1, the overlap is set to two
    print("Overlap computed")

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("Peak Memory usage in Merging step:", memory_usage, "GB")
    ###

    mask1 = None # Deallocate memory
    mask2 = None # Deallocate memory

    # Add the two registered images together
    merged_image = sitk.Add(sitk.Cast(moving_image, sitk.sitkUInt16), sitk.Cast(fixed_image, sitk.sitkUInt16)) #Use 16 bit because the sum might clip at 255
    print("Images stitched")

    ### Print Memory usage
    # Get the current process ID
    pid = psutil.Process()
    # Get the memory usage in bytes
    memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
    # Print the memory usage in gigabytes
    print("2nd Peak Memory usage in Merging step:", memory_usage, "GB")
    ###

    fixed_image = None # Deallocate memory
    moving_image = None # Deallocate memory

    overlap_image = sitk.GetImageFromArray(overlap)
    overlap = None # Deallocate memory
    print("\nOverlap size: ", overlap_image.GetSize())
    print("Overlap dtype: ", overlap_image.GetPixelIDTypeAsString())
    overlap_image = sitk.Cast(overlap_image, sitk.sitkUInt16)

    # Divide the sum by the count to obtain the average
    merged_image = sitk.Divide(merged_image, overlap_image)
    overlap_image = None # Deallocate memory
    merged_image = sitk.Cast(merged_image, sitk.sitkUInt8)

    print("New Merged Image Size:", merged_image.GetSize())
    print("Pixel Type:", merged_image.GetPixelIDTypeAsString())
    print("Merging Images Done!\n")

    return(merged_image, canvas)


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

#mask_00 = mask2mask(fixed_mask_path)
#mask_01 = mask2mask(moving_mask_path)

#print(mask_00.GetSize())
#print(np.unique(sitk.GetArrayFromImage(mask_00)))
#print(mask_00.GetPixelIDTypeAsString())

#print("")

#print(mask_01.GetSize())
#print(np.unique(sitk.GetArrayFromImage(mask_01)))
#print(mask_01.GetPixelIDTypeAsString())

# Start a timer
start_time = time.time()

# Load the fixed and moving images
image_00 = sitk.ReadImage(fixed_image_path)
# Set the image spacing, origin, and direction if needed
image_00.SetSpacing((1.0, 1.0, 1.0))  # Adjust spacing according to your image
image_00.SetOrigin((0.0, 0.0, 0.0))  # Adjust origin according to your image
image_00.SetDirection(np.eye(3).flatten())  # Adjust direction according to your image
print("Loading first image done")

image_01 = sitk.ReadImage(moving_image_path)
# Set the image spacing, origin, and direction if needed
image_01.SetSpacing((1.0, 1.0, 1.0))  # Adjust spacing according to your image
image_01.SetOrigin((0.0, 0.0, 0.0))  # Adjust origin according to your image
image_01.SetDirection(np.eye(3).flatten())  # Adjust direction according to your image
print("Loading second image done")

# Get the current process ID
pid = psutil.Process()
# Get the memory usage in bytes
memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
# Print the memory usage in gigabytes
print("Memory usage after loading the data:", memory_usage, "GB")

########################################################################
##### Optional: Mask the images with segmentation for registration #####
#image_00 = sitk.Mask(image_00, mask_00, maskingValue=0, outsideValue=0)
#image_01 = sitk.Mask(image_01, mask_01, maskingValue=0, outsideValue=0)
########################################################################

# Registration with no downsampling!
final_transform = four_step_registration(image_00, image_01, z_shift, 1)

print("\nFinal transform:", final_transform)

average_image, canvas = merge_ct(image_00, image_01, final_transform)

### Print Memory usage
# Get the current process ID
pid = psutil.Process()
# Get the memory usage in bytes
memory_usage = (pid.memory_info().rss) / (1024 * 1024 * 1024)
# Print the memory usage in gigabytes
print("\nMemory usage merging:", memory_usage, "GB")


#######################################################
##### Optional: Mask the images with segmentation #####
#average_image_clean = mask2crop(average_image, canvas, final_transform, mask_00, mask_01)
#print(average_image_clean.GetPixelIDTypeAsString())
#######################################################
sitk.WriteImage(average_image, OUTPUT)
#sitk.WriteImage(average_image_clean, OUTPUT)

# End the timer
end_time = time.time()
# Calculate the execution time in seconds
execution_time = end_time - start_time
# Print the execution time
print("\nExecution time:", execution_time, "seconds")
