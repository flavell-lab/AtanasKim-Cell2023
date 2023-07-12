import sys
import csv
import os
import h5py
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import cluster, signal, ndimage, spatial, io, ndimage
from functools import reduce

from skimage import measure, filters
from timeit import default_timer as timer

fixed_path = sys.argv[1]
moving_path = sys.argv[2]
tfm_path = sys.argv[3]
# Formatted as x,y
fixed_head= sys.argv[4]
# Formatted as x,y
moving_head = sys.argv[5]
try:
    fixed_mask_path = sys.argv[5]
    fixed_image_mask = sitk.ReadImage(fixed_mask_path)
except:
    print("No mask provided")

moving_idx = os.path.basename(sys.argv[2])[0:-4]

print("Processing moving image: ", moving_path)

euler_shrink = [4, 2]
euler_smooth_sig = [3, 2] # smoothing sigma for Gaussian filter for multi-resolution reg
euler_sample_perc= [0.3, 0.3] # sample percentage for multi-resolution reg

def unpack_tuple(parameter):
    return reduce(lambda x, y: str(x) + " " + str(y), parameter)

def img_norm(img):
    return (img - np.mean(img)) / np.std(img)

def shrink_metric_img(img, tile_size=20):
    img_size = int(400 / tile_size)
    return_img = np.zeros((img_size, img_size))
    for x in range(0, img_size):
        for y in range(0, img_size):
            return_img[x, y] = np.mean(img[tile_size * x:tile_size * x + tile_size, tile_size * y:tile_size * y + tile_size])
    return return_img

def cart_to_polar(vector):
    x = vector[0]
    y = vector[1]
    return (x ** 2 + y ** 2) ** 0.5, np.arctan2(y,x)

def vector_angle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))

    if sinang >= 0:
        return np.arctan2(sinang, cosang)
    else:
        return 2 * np.pi - np.arctan2(sinang, cosang)
    
def calculate_dist(pt1, pt2):
    return ((pt1[0]-pt2[0]) ** 2 + (pt1[1]-pt2[1]) ** 2) ** 0.5

def calc_centroid(img, idx_list):
    total_mass = 0.0
    vector_sum = [0.0, 0.0]
    for idx_x, idx_y in idx_list:
        mass = img[idx_x, idx_y]
        total_mass += mass
        vector_sum[0] += mass * idx_x
        vector_sum[1] += mass * idx_y
    return [vector_sum[i] / total_mass for i in range(len(vector_sum))]

def find_key_points(component_list, img_centroid):
    list_len = len(component_list)
    dist_array = np.zeros((list_len, list_len))
    
    for i in range(0, list_len):
        for j in range(0, list_len):
            dist_array[i, j] = calculate_dist(component_list[i].centroid, component_list[j].centroid)

    c1, c2 = np.unravel_index(np.argmax(dist_array), dist_array.shape)
    total_dist = np.sum(dist_array, 0)
    
    c1_dist = calculate_dist(img_centroid, component_list[c1].centroid)
    c2_dist = calculate_dist(img_centroid, component_list[c2].centroid)
    
    # returning head and tail
    if c1_dist < c2_dist:
        return component_list[c1].centroid, component_list[c2].centroid
    else:
        return component_list[c2].centroid, component_list[c1].centroid
        
def report_progress(iter_list, desc: str, metric, learn, verbose=True):
    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    iter_list.append([desc, time_string, metric, learn])
    if verbose:
        print_statement = time_string + " : " + desc + " - metric: " + str(metric) + " learn: " + str(learn)
        print(print_statement)

def add_report_cmd(reg, iter_list, verbose=True):
    reg.AddCommand(sitk.sitkIterationEvent,
                    lambda: report_progress(iter_list, 'iteration', reg.GetMetricValue(), reg.GetOptimizerLearningRate(), verbose))
    reg.AddCommand(sitk.sitkStartEvent,
                    lambda: report_progress(iter_list, 'registration start', reg.GetMetricValue(), reg.GetOptimizerLearningRate(), verbose))
    reg.AddCommand(sitk.sitkEndEvent,
                    lambda: report_progress(iter_list, 'registration end', reg.GetMetricValue(), reg.GetOptimizerLearningRate(), verbose))
    reg.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                    lambda: report_progress(iter_list, 'changing resolution', reg.GetMetricValue(), reg.GetOptimizerLearningRate(), verbose))

def add_append_tfm_data(reg, tfm_iter_list):
    reg.AddCommand(sitk.sitkIterationEvent, lambda: tfm_iter_list.append(reg.GetOptimizerPosition()))

def reg_metric_compare(fixed, moving, tfm, kernel_size=2, interpolator=sitk.sitkLinear, verbose=True):
    reg_metric = sitk.ImageRegistrationMethod()
    reg_metric.SetMetricAsCorrelation()
    reg_metric.SetInterpolator(interpolator)
    
    before = reg_metric.MetricEvaluate(fixed, moving)
    after = reg_metric.MetricEvaluate(sitk.DiscreteGaussian(fixed, kernel_size), sitk.DiscreteGaussian(moving, kernel_size))
    reg_metric.SetMovingInitialTransform(tfm)
    before_gaussian = reg_metric.MetricEvaluate(fixed, moving)
    after_gaussian = reg_metric.MetricEvaluate(sitk.DiscreteGaussian(fixed, kernel_size), sitk.DiscreteGaussian(moving, kernel_size))
    
    if verbose:
        print("Before:", before)
        print("After:", after)
        print("Before (gaussian):", before_gaussian)
        print("After (gaussian):", after_gaussian)
        
    return before, after, before_gaussian, after_gaussian

#### Import
fixed_image = sitk.ReadImage(fixed_path)
moving_image = sitk.ReadImage(moving_path)

fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

#### Initial angle and displacement calculation
fixed_image_spacing = fixed_image.GetSpacing()

fixed_mip_norm = np.max(sitk.GetArrayFromImage(sitk.Normalize(sitk.DiscreteGaussian(fixed_image, 1))) > 6, 0)
moving_mip_norm = np.max(sitk.GetArrayFromImage(sitk.Normalize(sitk.DiscreteGaussian(moving_image, 1))) > 6, 0)
fixed_mip_seg = measure.label(fixed_mip_norm, background=0)
moving_mip_seg = measure.label(moving_mip_norm, background=0)
fixed_mip_seg_component = measure.regionprops(fixed_mip_seg, np.max(sitk.GetArrayFromImage(sitk.DiscreteGaussian(fixed_image, 1)), 0))
moving_mip_seg_component = measure.regionprops(moving_mip_seg, np.max(sitk.GetArrayFromImage(sitk.DiscreteGaussian(moving_image, 1)), 0))

fixed_global_centroid = np.array(calc_centroid(np.max(sitk.GetArrayFromImage(fixed_image), 0), np.argwhere(fixed_mip_norm == True)))
print(fixed_global_centroid)
moving_global_centroid = np.array(calc_centroid(np.max(sitk.GetArrayFromImage(moving_image), 0), np.argwhere(moving_mip_norm == True)))
print(moving_global_centroid)



# fixed_head, fixed_tail = find_key_points(fixed_mip_seg_component, fixed_global_centroid)
# moving_head, moving_tail =  find_key_points(moving_mip_seg_component, moving_global_centroid)
fixed_head = np.array([float(x) for x in str.split(fixed_head, ",")])
moving_head = np.array([float(x) for x in str.split(moving_head, ",")])
print(fixed_head)
print(moving_head)

# Python's coordinates are opposite of Fiji/Julia, so swap x and y in fixed head input
v_fixed = (fixed_head[1] - fixed_global_centroid[0], fixed_head[0] - fixed_global_centroid[1])
v_moving = (moving_head[1] - fixed_global_centroid[0], moving_head[0] - moving_global_centroid[1])

# center of image
center = np.array(sitk.GetArrayFromImage(fixed_image).shape[1:3])/2
# centroid of moving worm relative to center
moving_delta = moving_global_centroid[0:2] - center
# angle between two images
target_angle = np.arctan2(v_fixed[1], v_fixed[0]) - np.arctan2(v_moving[1], v_moving[0])
# transform from fixed to moving frame based on angle
transform = np.matrix([[np.cos(target_angle), -np.sin(target_angle)], [np.sin(target_angle), np.cos(target_angle)]])
# transformed moving centroid to fixed image coordinates
transformed_moving_centroid = np.dot(transform, moving_delta) + center
print(transformed_moving_centroid)
# vector from fixed to moving centroid, transformed back into moving image coordinates
target_delta = np.dot(np.linalg.inv(transform), np.transpose(transformed_moving_centroid - fixed_global_centroid[0:2]))
target_delta_x = target_delta[0,0]#fixed_global_centroid[0] - transformed_moving_x_centroid
target_delta_y = target_delta[1,0]#fixed_global_centroid[1] - transformed_moving_y_centroid
# TODO: target_delta_z

print("Target angle:", target_angle) 
print("Target delta_x:", target_delta_x)
print("Target delta_y:", target_delta_y)


#### Euler repeats 
def euler():
    registration_method = sitk.ImageRegistrationMethod()

    euler_tfm_init = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                   sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    euler_parm = list(euler_tfm_init.GetParameters())
    euler_parm[2] = target_angle
    euler_parm[3] = target_delta_y * fixed_image_spacing[0]
    euler_parm[4] = target_delta_x * fixed_image_spacing[0]

    euler_tfm=sitk.Euler3DTransform()
    euler_tfm.SetFixedParameters(euler_tfm_init.GetFixedParameters())
    euler_tfm.SetParameters(euler_parm)

    return euler_tfm

euler_tfm_final_list = []

euler_repeat = 1
euler_after_g_min = 0 
euler_min_idx = 0

for i in range(0, euler_repeat):
    print("Euler round: " + str(i))
    tfm_final = euler()
    euler_before, euler_before_g, euler_after, euler_after_g = reg_metric_compare(fixed_image, moving_image, tfm_final, kernel_size=2, interpolator=sitk.sitkLinear, verbose=False)
    
    print("Euler (gaussian after)" + str(euler_after_g))

    if euler_after_g < euler_after_g_min:
        print("New min found")
        euler_min_idx = i
        euler_after_g_min = euler_after_g
    
    euler_tfm_final_list.append(tfm_final)
    
    if euler_after_g < -0.8 and 2 < i:
        break

print("Euler best idx: " + str(euler_min_idx) + "\n")
        
euler_tfm = euler_tfm_final_list[euler_min_idx]

#### Writing preview
moving_resampled = sitk.Resample(moving_image, fixed_image, euler_tfm, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
imc1=np.max(sitk.GetArrayFromImage(fixed_image),0)
imc2=np.max(sitk.GetArrayFromImage(moving_resampled),0)
imc3=np.zeros(imc1.shape)
imc4=np.max(sitk.GetArrayFromImage(moving_image),0)
plt.imsave(os.path.join(os.path.dirname(tfm_path), moving_idx + ".png"), np.swapaxes(np.stack((imc1/np.max(imc1), imc2/np.max(imc2), imc3)),2,0))
plt.imsave(os.path.join(os.path.dirname(tfm_path), moving_idx + "_moving.png"), np.swapaxes(imc4/np.max(imc4),1,0))

#### Writing transform file
parameter_txt = "(Transform \"EulerTransform\")" + "\n" \
"(NumberOfParameters 6)" + "\n" \
"(TransformParameters " + unpack_tuple(euler_tfm.GetParameters()) + ")" + "\n" \
"(InitialTransformParametersFileName \"NoInitialTransform\")" + "\n" \
"(UseBinaryFormatForTransformationParameters \"false\")" + "\n" \
"(HowToCombineTransforms \"Compose\")" + "\n" \
"// Image specific" + "\n" \
"(FixedImageDimension 3)" + "\n" \
"(MovingImageDimension 3)" + "\n" \
"(FixedInternalImagePixelType \"float\")" + "\n" \
"(MovingInternalImagePixelType \"float\")" + "\n" \
"(Size " + unpack_tuple(fixed_image.GetSize())+ ")" + "\n" \
"(Index 0 0 0)" + "\n" \
"(Spacing " + unpack_tuple(fixed_image.GetSpacing()) + ")" + "\n" \
"(Origin " + unpack_tuple(fixed_image.GetOrigin()) + ")" + "\n" \
"(Direction " + unpack_tuple(fixed_image.GetDirection()) + ")" + "\n" \
"(UseDirectionCosines \"true\")" + "\n" \
"// EulerTransform specific" + "\n" \
"(CenterOfRotationPoint " + unpack_tuple(euler_tfm.GetFixedParameters()) + ")" + "\n" \
"(ComputeZYX \"false\")" + "\n" \
"// ResampleInterpolator specific" + "\n" \
"(ResampleInterpolator \"FinalBSplineInterpolator\")" + "\n" \
"(FinalBSplineInterpolationOrder 3)" + "\n" \
"// Resampler specific" + "\n" \
"(Resampler \"DefaultResampler\")" + "\n" \
"(DefaultPixelValue 0.000000)" + "\n" \
"(ResultImageFormat \"nrrd\")" + "\n" \
"(ResultImagePixelType \"short\")" + "\n" \
"(CompressResultImage \"true\")"

with open(tfm_path, "w") as f:
    f.write(parameter_txt)
