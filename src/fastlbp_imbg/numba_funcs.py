# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:54:29 2021

@author: bw17
"""


from numba import njit, prange
import numpy as np
import random
import math

@njit
def rgb2labelint_iterator(img, array_of_colors):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    len_array_colors = len(array_of_colors)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(len_array_colors):
                if np.all(np.equal(array_of_colors[k], img[i,j])):
                    output[i,j] = k
                    continue
#            output[i,j] = np.argmax(np.all(np.equal(img[i,j], array_of_colors), axis=1))
    return output

@njit(parallel=True)
def rgb2commonrgbclosest_iterator(img, array_of_colors):
    """
    This function forces colors into the nearest in the array of 
    colors (rgb or rgba)

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    array_of_colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
    len_array_colors = len(array_of_colors)
    for i in prange(img.shape[0]):
        for j in range(img.shape[1]):
            distances = np.zeros(len_array_colors, dtype=np.float64)
            for k in range(len_array_colors):
                this_distance_sq = float(0)
                for m in range(img.shape[2]):
                    this_distance_sq += (img[i,j,m]-array_of_colors[k,m])**2
                distances[k] = np.sqrt(this_distance_sq)
                output[i,j] = array_of_colors[np.argmin(distances)]
#            output[i,j] = np.argmax(np.all(np.equal(img[i,j], array_of_colors), axis=1))
    return output

@njit(parallel=True)
def rgb2commonrgbclosest_iterator_with_min_distance(img, array_of_colors):
    """
    This function forces colors into the nearest in the array of 
    colors (rgb or rgba)

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    array_of_colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
    output_distance = np.zeros((img.shape[0], img.shape[1]), dtype=np.float)
    len_array_colors = len(array_of_colors)
    for i in prange(img.shape[0]):
        for j in range(img.shape[1]):
            distances = np.zeros(len_array_colors, dtype=np.float64)
            for k in range(len_array_colors):
                this_distance_sq = float(0)
                for m in range(img.shape[2]):
                    this_distance_sq += (img[i,j,m]-array_of_colors[k,m])**2
                distances[k] = np.sqrt(this_distance_sq)
                output[i,j] = array_of_colors[np.argmin(distances)]
                output_distance[i,j] = np.min(distances)
    return output, output_distance

@njit
def upsize(array, scaling):
    scale = round(scaling)
    x0_test = array.shape[0]*scale
    x1_test = array.shape[1]*scale

    x0 = round(x0_test)
    x1 = round(x1_test)
    if  x0-x0_test > 0.0001:
        print('Array dimensions must divide by scaling')
    if  x1-x1_test > 0.0001:
        print('Array dimensions must divide by scaling')
        
    if len(array.shape) == 2:
        outarray = np.zeros((x0, x1), dtype = array.dtype)
        
        patch = np.ones((scale, scale), dtype = array.dtype)
        
        for i in range(0, array.shape[0]):
    #        update_progress(i/array.shape[0])
            for j in range(0, array.shape[1]):
                outarray[scale*i:scale*(i+1), scale*j:scale*(j+1)] = array[i,j] * patch
    
    return outarray

@njit(parallel=True)
def upsize2(array, scaling):
    scale = round(scaling)
    x0_test = array.shape[0]*scale
    x1_test = array.shape[1]*scale
    x0 = round(x0_test)
    x1 = round(x1_test)
    if  x0-x0_test > 0.0001:
        print('Array dimensions must divide by scaling')
    if  x1-x1_test > 0.0001:
        print('Array dimensions must divide by scaling')
        
    if len(array.shape) == 3:
        outarray = np.zeros((x0, x1, round(array.shape[2])), dtype = array.dtype)
        
        patch = np.ones((scale, scale, round(array.shape[2])), dtype = array.dtype)
        
        for i in prange(0, array.shape[0]):
    #        update_progress(i/array.shape[0])
            for j in prange(0, array.shape[1]):
                outarray[scale*i:scale*(i+1), scale*j:scale*(j+1), :] = array[i,j] * patch
    
    return outarray

@njit
def get_euclidean_distance(np_vec1, np_vec2):
    length = len(np_vec1)
    total = 0
    for i in range(length):
        total += (np_vec1[i]-np_vec2[i])**2
    return np.sqrt(total)


@njit
def change_classes_numpy(array, arraylist0, arraylist1):
    #remaps the classes in an image to new values in arraylist1
    outarray = array.copy()
    iterarray = np.stack((arraylist0,arraylist1)).T
    for each0, each1 in iterarray:
        for i in range(0, array.shape[0]):
            for j in range(0, array.shape[1]):
                if array[i,j] == each0:
                    outarray[i,j] = each1
    return outarray

@njit
def IoU_labels_images(calculated_labels_image, groundtruth_labels_image):
    #assumes the images are the same size
    h = groundtruth_labels_image.shape[0]
    w = groundtruth_labels_image.shape[1]
    intersection = 0
    for i in range(0,h):
        for j in range(0,w):
            if calculated_labels_image[i,j] == groundtruth_labels_image[i,j]:
                intersection += 1
    return intersection/(h*w)

@njit
def IoU_labels_images_1d(calculated_labels_image, groundtruth_labels_image):
    #assumes the images are the same size
    h = groundtruth_labels_image.shape[0]
    intersection = 0
    for i in range(0,h):
        if calculated_labels_image[i] == groundtruth_labels_image[i]:
                intersection += 1
    return intersection/h


@njit
def simulated_annealing_on_clusters(calculated_labels_arraylist_initial, groundtruth_labels_arraylist, 
                                    calculated_labels_image, groundtruth_labels_image,
                                    initial_temp=90, final_temp=0.1, alpha=0.001):
    
    def get_success_score(current_state):
        """Calculates cost of the argument state for your solution."""
        remap_labels_image = change_classes_numpy(calculated_labels_image, current_state, groundtruth_labels_arraylist) 
        return IoU_labels_images(remap_labels_image, groundtruth_labels_image)
    
    def get_random_neighbor(current_state):
        x1, x2 = np.random.randint(0, len(current_state), size=2)
        neighbor = current_state.copy()
        neighbor[x2] = current_state[x1]
        neighbor[x1] = current_state[x2]
        """Returns neighbors of the argument state for your solution."""
        return neighbor
    
    """Peforms simulated annealing to find a solution"""   
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = calculated_labels_arraylist_initial
    best_state = calculated_labels_arraylist_initial.copy()
    best_state_success_score = get_success_score(best_state)

    while current_temp > final_temp:
        neighbor = get_random_neighbor(current_state)

        # Check if neighbor is best so far
        current_state_success_score = get_success_score(current_state)
        success_score_diff = get_success_score(neighbor) - current_state_success_score

        # if the new solution is better, accept it
        if success_score_diff > 0:
            current_state = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) > math.exp(-success_score_diff / current_temp):
                current_state = neighbor
                
        if current_state_success_score > best_state_success_score:
            best_state = current_state.copy()
            best_state_success_score = float(current_state_success_score)
                
        # decrement the temperature
        current_temp -= alpha
    
    return best_state, best_state_success_score



@njit
def test_on_edge(shape, i,j,k):
    #for a 3D array, find whether an element is on the edge of it
    bool0 = i==0 or i == shape[0]-1
    bool1 = j==0 or j == shape[1]-1
    bool2 = k==0 or k == shape[2]-1
    
    edge = (bool0 & bool1) or (bool1 & bool2) or (bool0 & bool2)
    return edge

@njit
def create_3D_wireframe_edge(shape):
    #create a 1 pixel wireframe for the given shape
    my_wireframe = np.zeros(shape, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if test_on_edge(shape, i, j, k):
                    my_wireframe[i,j,k] = 1
    return my_wireframe


@njit
def extract_masked_area(img, mask, xycoords=True):
    #img needs to be 3D even if only one dimension is only 1 length
    #mask needs to be 1D with zeros and ones
    output_length = np.sum(mask)
    output = np.zeros((output_length, img.shape[2]), dtype=img.dtype)
    output_X0 = np.zeros(output_length, dtype='i8') #
    output_X1 = np.zeros(output_length, dtype='i8')
    #print(output_X1.dtype)
    count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 1:
                output[count] = img[i,j]
                output_X0[count] = i
                output_X1[count] = j
                count += 1
    return output, output_X0, output_X1

@njit
def get_object_image_by_indices(labeled, object_indices):
    """
    This function takes a labeled image (labelled with ndi.label for example) 
    so that the pixels have the values of different labels.
    labeled is a numpy array of integers
    object_indices is a numpy array of the integers of the objects desired to extract
    returns an output image of the same shape as labeled that is a bool image
    """
    myshape = labeled.shape
    output = np.zeros(myshape, dtype = np.bool_)
    for i in range(myshape[0]):
        for j in range(myshape[1]):
            if labeled[i,j] in object_indices:
                output[i,j] = 1
                
    return output


@njit
def chi_sq(p0, p1):
    """
    the chi_sq returns the chi squared distance between two vectors
    both of them are on the bottom of the denominator

    Parameters
    ----------
    p0 : 1D numpy array
        One of the vectors to find distance between
    p1 : 1D numpy array
        The second vector to 

    Returns
    -------
    total : Chi squared distance
        DESCRIPTION.

    """
    total = 0
    for i in range(len(p0)):
        if p0[i] == 0 and p1[i] == 0:
            continue
        else:
            to_add = ((p0[i]-p1[i])**2)/(p0[i] + p1[i])
            total += to_add        
    return total


@njit
def chi_sq_abs_denominator(p0, p1):
    """
    the chi_sq returns the chi squared distance between two vectors
    both of them are on the bottom of the denominator

    Parameters
    ----------
    p0 : 1D numpy array
        One of the vectors to find distance between
    p1 : 1D numpy array
        The second vector to 

    Returns
    -------
    total : Chi squared distance
        DESCRIPTION.

    """
    total = 0
    for i in range(len(p0)):
        if p0[i] == 0 and p1[i] == 0:
            continue
        elif p0[i]+p1[i] == 0:
            continue
        else:
            to_add = ((p0[i]-p1[i])**2)/np.abs(p0[i] + p1[i])
            total += to_add
    return total


@njit
def get_edges_of_cluster_shapes(intlabels, rgblabels, k = 2):
    """
    This function takes labels in two formats and returns the 
    label edges as a k-pixel edge image

    Parameters
    ----------
    intlabels : numpy array of ints with bg=0
        Labels in the format of an image with a different int for
        each label
    rgblabels : numpy array of floats RGB
        after from skimage.color import label2rgb has been called 
        on intlabels, with no image behind it
    k : int
        k is basically the thickness of the edging#
    output_vals : 'rgb' or 'ints'

    Returns
    -------
    output_edges 
        the image of output edges between the labels as rgb

    """
    

    shape = rgblabels.shape
    output_edges = np.zeros(shape, rgblabels.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            this_val = intlabels[i,j]
            if this_val == 0:
                continue
            else:
#                compare_array = np.array([intlabels[i-1,j], intlabels[i+1,j], 
#                                          intlabels[i,j-1], intlabels[i,j+1]])

                compare_array = intlabels[i-k:i+k+1,j-k:j+k+1]
                allsame = np.all(compare_array == this_val)
                if allsame:
                    continue
                else:
                    output_edges[i,j] = rgblabels[i,j]
        
    return output_edges

@njit
def get_edges_of_cluster_shapes_int_output(intlabels, rgblabels, k = 2):
    """
    This function takes labels in two formats and returns the 
    label edges as a k-pixel edge image

    Parameters
    ----------
    intlabels : numpy array of ints with bg=0
        Labels in the format of an image with a different int for
        each label
    rgblabels : numpy array of floats RGB
        after from skimage.color import label2rgb has been called 
        on intlabels, with no image behind it
    k : int
        k is basically the thickness of the edging#
    output_vals : 'rgb' or 'ints'

    Returns
    -------
    output_edges 
        the image of output edges between the labels as ints

    """

    shape = intlabels.shape
    output_edges = np.zeros(shape, intlabels.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            this_val = intlabels[i,j]
            if this_val == 0:
                continue
            else:
#                compare_array = np.array([intlabels[i-1,j], intlabels[i+1,j], 
#                                          intlabels[i,j-1], intlabels[i,j+1]])

                compare_array = intlabels[i-k:i+k+1,j-k:j+k+1]
                allsame = np.all(compare_array == this_val)
                if allsame:
                    continue
                else:
                    output_edges[i,j] = intlabels[i,j]
        
    return output_edges

@njit
def get_edges_of_cluster_shapes_with_image_edges(intlabels, rgblabels, k = 2):
    """
    This function takes labels in two formats and returns the 
    label edges as a k-pixel edge image

    Parameters
    ----------
    intlabels : numpy array of ints with bg=0
        Labels in the format of an image with a different int for
        each label
    rgblabels : numpy array of floats RGB
        after from skimage.color import label2rgb has been called 
        on intlabels, with no image behind it
    k : int
        k is basically the thickness of the edging

    Returns
    -------
    output_edges 
        the image of output edges between the labels

    """
    
    shape = rgblabels.shape
    output_edges = np.zeros(shape, rgblabels.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            this_val = intlabels[i,j]
            if this_val == 0:
                continue
            else:
                if i<k or shape[0]-i<k or j<k or shape[1]-j<k:
                    output_edges[i,j] = rgblabels[i,j]
                else:
                    compare_array = intlabels[i-k:i+k+1,j-k:j+k+1]
                    allsame = np.all(compare_array == this_val)
                    
                    if allsame:
                        continue
                    else:
                        output_edges[i,j] = rgblabels[i,j]
    return output_edges

@njit
def get_edges_of_cluster_shapes_with_image_edges_int_output(intlabels, rgblabels, k = 2):
    """
    This function takes labels in two formats and returns the 
    label edges as a k-pixel edge image

    Parameters
    ----------
    intlabels : numpy array of ints with bg=0
        Labels in the format of an image with a different int for
        each label
    rgblabels : numpy array of floats RGB
        after from skimage.color import label2rgb has been called 
        on intlabels, with no image behind it
    k : int
        k is basically the thickness of the edging

    Returns
    -------
    output_edges 
        the image of output edges between the labels as int output

    """
    
    shape = intlabels.shape
    output_edges = np.zeros(shape, intlabels.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            this_val = intlabels[i,j]
            if this_val == 0:
                continue
            else:
                if i<k or shape[0]-i<k or j<k or shape[1]-j<k:
                    output_edges[i,j] = intlabels[i,j]
                else:
                    compare_array = intlabels[i-k:i+k+1,j-k:j+k+1]
                    allsame = np.all(compare_array == this_val)
                    
                    if allsame:
                        continue
                    else:
                        output_edges[i,j] = intlabels[i,j]
    return output_edges



def get_edges_of_cluster_shapes_with_partial_upscale(intlabels, rgblabels, k=2, 
                                                     partial_upscale=1, final_upscale=1,
                                                     image_edges_drawn=False, output_type='rgb'):
    """
    A function to turn an image of block clusters into edged clusters.
    This allows the inside of each cluster to be seen. 
    Output may need to be converted to uint8 to be displayed with napari.
    For example
    from skimage import img_as_ubyte
    x_ubyte = img_as_ubyte(x)
    To display properly with napari, you will also need to run
    convert_rgb_to_rgba, with 
     transparent=np.array([0,0,0]), transparency_val = int(255)
     Then blending translucent
          Showing stuff clearly on napari also seems to require the background 
     to be =1 not =0

    Parameters
    ----------
    intlabels : numpy array of ints with bg=0
        Labels in the format of an image with a different int for
        each label
    rgblabels : numpy array of floats RGB
        after from skimage.color import label2rgb has been called 
        on intlabels, with no image behind it
    k : int
        k is basically the thickness of the edging
    partial_upscale : int, optional
        How much to increase the size of the image before drawing the boundary
        lines The default is 1.
    final_upscale : int, optional
        The final size of the output image. Probably the same as the 
        patchsize. The default is 1.
    image_edges_drawn : TYPE, optional
        Choose whether to have the edges that encounter the edge of the image
        drawn in or not. Useful when the area under study goes right up to the 
        edges. The default is False.

    Returns
    -------
    output_edges_full.
    Output may need to be converted to uint8 to be displayed with napari.
    For example
    from skimage import img_as_ubyte
    x_ubyte = img_as_ubyte(x)
    To display properly with napari, you will also need to run
    convert_rgb_to_rgba, with 
     transparent=np.array([0,0,0]), transparency_val = int(255)
     Then blending translucent
     Showing stuff clearly on napari also seems to require the background 
     to be =1 not =0
    """
    intlabels_partial = upsize(intlabels, partial_upscale)
    rgblabels_partial = upsize2(rgblabels, partial_upscale)
#    print(intlabels_partial)
    if image_edges_drawn and output_type=='rgb':
        output_edges_partial = get_edges_of_cluster_shapes_with_image_edges(intlabels_partial, 
                                                                            rgblabels_partial, k)
    elif not image_edges_drawn and output_type=='rgb':
        output_edges_partial = get_edges_of_cluster_shapes(intlabels_partial,
                                                           rgblabels_partial, k)
    elif image_edges_drawn and output_type=='int':
         output_edges_partial = get_edges_of_cluster_shapes_with_image_edges_int_output(intlabels_partial, 
                                                                            rgblabels_partial, k)
    elif not image_edges_drawn and output_type=='int':
         output_edges_partial = get_edges_of_cluster_shapes_int_output(intlabels_partial,
                                                           rgblabels_partial, k)
        
        
    further_upscale = int(final_upscale/partial_upscale)
    
    if len(output_edges_partial.shape) == 2:
        output_edges_full = upsize(output_edges_partial, further_upscale)
    elif len(output_edges_partial.shape) == 3:
        output_edges_full = upsize2(output_edges_partial, further_upscale)
    
    return output_edges_full

@njit
def convert_rgb_to_rgba(img, transparent=np.array([0.,0.,0.]), transparency_val = int(1)):
    """
    Converts rgb to rgba while setting exactly one colur to fully transparent
    and preserving the rest.
    
    There is a useful func version of this that just sets one transparency
    
        To display properly with napari, you will also need to run
    convert_rgb_to_rgba, with 
     transparent=np.array([0,0,0]), transparency_val = int(255)
     Then blending translucent
          Showing stuff clearly on napari also seems to require the background 
     to be =1 not =0

    Parameters
    ----------
    img : numpy array (of floats i think, rgb)
        DESCRIPTION.
    transparent : numpy array of a 3 vec rgb, optional
        This is the colour which becomes transparent
        The default is np.array([0.,0.,0.]), which is black.
    transparency_val is for the non-transparent values

    Returns
    -------
    rgba_image : numpy array like img 
        RGBA version of img

    """
    shape = img.shape
    rgba_image = np.zeros((shape[0], shape[1], 4), img.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.array_equal(img[i,j], transparent):
                rgba_image[i,j, 0:4] = 0
            else:
                rgba_image[i,j, 0:3] = img[i,j]
                rgba_image[i,j,3] = transparency_val
    return rgba_image

@njit
def set_transparent_color_in_rgba(img, transparent=np.array([0., 0., 0.])):
    """
    A function to take a numpy array image rgba (4 channel) and set one of 
    the colors to clear. Mostly used for overlaying a mask onto another image. 

    Parameters
    ----------
    img : numpy array
        An rgba (4 channels) image that needs to have one color made 
        transparent
    transparent : numpy array, optional
        The color that is to be made transparent. 
        The default is np.array([0., 0., 0.]).

    Returns
    -------
    rgba_out_image : numpy array
        Same image but with one color made transparent.

    """
    shape = img.shape
    rgba_out_image = np.zeros((shape[0], shape[1], 4), img.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.array_equal(img[i,j, 0:3], transparent):
                rgba_out_image[i,j, 0:4] = 0
            else:
                rgba_out_image[i,j, 0:4] = img[i,j]
    return rgba_out_image

@njit
def convert_mask_to_rgba(mask, transparent=False, transparency_val = int(255)):
    """
    Converts mask to rgba while setting exactly one colur to fully transparent
    and preserving the other color.
    
    There is a useful func version of this that just sets one transparency

    Parameters
    ----------
    img : numpy array (of floats i think, rgb)
        DESCRIPTION.
    transparent : numpy array of either True or False, optional
        This is the colour which becomes transparent
        The default is False, which is black.

    Returns
    -------
    rgba_image : numpy array like img 
        RGBA version of img
        Use blending transluncent

    """
    shape = mask.shape
    rgba_image = np.zeros((shape[0], shape[1], 4), np.ubyte)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.array_equal(mask[i,j], transparent):
                rgba_image[i,j, 0:4] = 0
            else:
                rgba_image[i,j, 0:3] = 0 #was mask[i,j]
                #change this above value to 1 if you want 
                #blocks of white
                #or could change a different 3 vec of rgb
                rgba_image[i,j,3] = transparency_val
    return rgba_image

@njit
def keep_only_specified_colors(mask, array_of_colors):
#    output = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=mask.dtype)
    output = mask.copy()
    len_array_colors = len(array_of_colors)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            found_a_match = 0
            for k in range(len_array_colors):
                if np.all(np.equal(array_of_colors[k], mask[i,j])):
                    found_a_match = 1
            if found_a_match == 0:
                output[i,j] = np.zeros(output.shape[2], dtype=output.dtype)
            else:
                continue
#            distances = np.zeros(len_array_colors, dtype=np.float64)
            
#            for k in range(len_array_colors):
#                this_distance_sq = float(0)
#                for m in range(mask.shape[2]):
#                    this_distance_sq += (mask[i,j,m]-array_of_colors[k,m])**2
#                distances[k] = np.sqrt(this_distance_sq)
#                output[i,j] = array_of_colors[np.argmin(distances)]
#            output[i,j] = np.argmax(np.all(np.equal(mask[i,j], array_of_colors), axis=1))
    return output


@njit(parallel=True)
def extract_color_count(image, color):
    """
    This function takes an image (rgb, rgba or ints) and counts the number
    of pixels where the specified color variable exists

    Parameters
    ----------
    image : numpy array
        normally rgb image
    color : numpy array, same as image
        Same as image.

    Returns
    -------
    output_count : int
        the number of pixels where the color can be found in the image

    """
    output_count = 0
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(np.equal(image[i,j], color)):
                output_count += 1
    return output_count

@njit
def extract_colors_count(image, colors):
    """
    This function gets the number of pixels for each of the colors provided
    
    Note: making this a numba function does not provide a speed up.
    Parameters
    ----------
    image : numpy array
        rgb or rgba or ints
    colors : numpy array 2D for rgb colors
        DESCRIPTION.

    Returns
    -------
    output_count : TYPE
        Frequency counts of the colors in hte image in the order the colors
        were provided

    """
    output_count = np.zeros(len(colors), dtype=np.int64)
    for i in range(len(colors)):
        output_count[i] = extract_color_count(image, colors[i])
    return output_count



@njit(parallel=True)
def extract_color_as_mask(image, color):
    """
    Select all of the pixels that match color and make an image mask out 
    of them.

    Parameters
    ----------
    image : numpy array
        DESCRIPTION.
    color : numpy array
        array of rgb values to match

    Returns
    -------
    output_mask : numpy array
        mask in the location of the color

    """
    output_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool_)
    for i in prange(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(np.equal(image[i,j], color)):
                output_mask[i,j] = 1
    return output_mask


@njit(parallel=True)
def rgb2label(img, array_of_colors, array_of_color_indices):
    """
    Converts rgb colors to numeric labels

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    array_of_colors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    array_of_colors = array_of_colors.astype(np.int64)
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.int64)
    len_array_colors = len(array_of_colors)
    for i in prange(img.shape[0]):
        for j in range(img.shape[1]):
            match = np.zeros(len_array_colors, dtype=np.uint8)
            for k in range(len_array_colors):
                match[k] = np.all(np.equal(img[i,j], array_of_colors[k]))
            output[i,j] = array_of_color_indices[np.argmax(match)]
    return output