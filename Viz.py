import os
import getopt
from common import *
import logging

import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime
import anndata as ad

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hdbscan
import umap
import skimage

import useful_functions as uf 
import numba_funcs as nf

from dataclasses import dataclass

# ==
from copy import deepcopy
os.environ['DISPLAY'] = ':1.0'
import napari
from matplotlib.colors import to_rgb, to_rgba
from skimage import img_as_ubyte
from numba import njit

# Use the novnc Desktop in the other window
# Note that this has to be run first, and separately to the rest of the code for it to work


@njit
def rgb2labelint_iterator(img, array_of_colors):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    len_array_colors = len(array_of_colors)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(len_array_colors):
                if np.all(np.equal(array_of_colors[k], img[i,j])):
                    output[i,j] = k
#            output[i,j] = np.argmax(np.all(np.equal(img[i,j], array_of_colors), axis=1))
    return output
    
    
def rgb2labelint(img, array_of_colors = None):
    if array_of_colors is None:
        array_of_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    output = rgb2labelint_iterator(img, array_of_colors)
    return output


@dataclass
class VizColors: 
    colors_with_inital_grey: 'typing.Any'
    rgb_colors_with_inital_grey: 'typing.Any'
    rgba_colors_with_inital_grey: 'typing.Any'
    rgba_colors_with_inital_transparent: 'typing.Any'
    dict_colors_initial_transp: 'typing.Any'
    dict_colors_initial_transp_0to1: 'typing.Any'
    dict_gray_out_colors: 'typing.Any'
    rgb_colors_with_inital_grey_0to1: 'typing.Any'
    df_colors: 'typing.Any'
    df_colors_out: 'typing.Any'
    df_colors_out_no_zero: 'typing.Any'

def load_viz_colors():
    colors_bundle = VizColors()

    #colors = uf.return_color_scale('Omer_chosen_colors_dark_first')
    colors_with_inital_grey = deepcopy(COLORS)
    colors_with_inital_grey.insert(0, 'gray')
    colors_bundle.colors_with_inital_grey = colors_with_inital_grey

    rgb_colors_with_inital_grey = np.asarray(255*np.array([to_rgb(color) for color in colors_with_inital_grey]), int)
    dict_colors = {i:color for i, color in enumerate(colors_with_inital_grey)}
    colors_bundle.rgb_colors_with_inital_grey = rgb_colors_with_inital_grey

    rgba_colors_with_inital_grey = np.asarray(255*np.array([to_rgba(color) for color in colors_with_inital_grey]), int)
    colors_bundle.rgba_colors_with_inital_grey = rgba_colors_with_inital_grey

    rgba_colors_with_inital_transparent = rgba_colors_with_inital_grey
    rgba_colors_with_inital_transparent[0] = 0
    dict_colors_initial_transp = {i:color for i, color in enumerate(rgba_colors_with_inital_transparent)}
    dict_colors_initial_transp_0to1 = {i:color/255 for i, color in enumerate(rgba_colors_with_inital_transparent)}
    colors_bundle.rgba_colors_with_inital_transparent = rgba_colors_with_inital_transparent
    colors_bundle.dict_colors_initial_transp_0to1 = dict_colors_initial_transp_0to1


    dict_gray_out_colors = {0:np.array([0,0,0,0], float), 1:np.array([0,0,0,1], float)}
    colors_bundle.dict_gray_out_colors = dict_gray_out_colors

    rgb_colors_with_inital_grey_0to1 = np.asarray(np.array([to_rgb(color) for color in colors_with_inital_grey]), float)
    colors_bundle.rgb_colors_with_inital_grey_0to1 = rgb_colors_with_inital_grey_0to1

    df_colors = uf.return_color_scale('block_colors_for_labels_against_white_small_points_white_first')
    df_colors_out = {i:each for i, each in enumerate(df_colors)}
    df_colors_out_no_zero = {key:value for key,value in df_colors_out.items() if key != 0}

    colors_bundle.df_colors = df_colors
    colors_bundle.df_colors_out = df_colors_out
    colors_bundle.df_colors_out_no_zero = df_colors_out_no_zero

    return colors_bundle


#
# ================= STAGE 5 - Napari Visualisation ====================
#

# uses env.patchsize, env.imgname, env.indir, env.outdir
def viz_load_single_data(input_file_name):
    TAG = 'viz_load_single_data'

    input_img_fullpath = os.path.join(env.indir, input_file_name)
    input_img_basename = safe_basename(input_img_fullpath)
    if not os.path.exists(input_img_fullpath):
        logging.error(f'{TAG}: Cannot find any original images in \'{env.indir}\' that look like {input_file_name}')
        sys.exit(1)

    logging.info(f'{TAG}: begin. input_img_basename={input_img_basename}')

    stage4_output_dir = get_outdir(STAGE4, input_img_basename)
    stage5_output_dir = get_outdir(STAGE5, input_img_basename)

    this_shape = get_dims_from_image(input_img_fullpath)
    image_width = this_shape[0]
    image_height = this_shape[1]

    directory_img = env.indir
    filename_img = input_file_name
    original_image = skimage.io.imread(input_img_fullpath)
    
    directory_output_masks = stage4_output_dir # 'outdir/img_basename/HDBScan_Output_masks/'
    filenames = os.listdir(directory_output_masks)
    if not filenames:
        logging.info(f'{TAG}: no hdb masks found at {directory_output_masks}. aborting.')
        sys.exit(1)
    logging.info(f'{TAG}: found hdb masks: {filenames} ; at {directory_output_masks}')

    this_index_filenames = [each for each in filenames if input_img_basename in each]
    logging.info(f'{TAG}: this_index_filenames: {this_index_filenames}')
    myfilename_ints = [each for each in this_index_filenames if 'ints' in each][0]
    logging.info(f'{TAG}: myfilename_ints: {myfilename_ints}')
    myfilename_rgb = [each for each in this_index_filenames if 'rgb' in each][0]
    logging.info(f'{TAG}: myfilename_rgb: {myfilename_rgb}')
    
    logging.info(f'{TAG}: begin loading data')
    intlabels = np.load(directory_output_masks + myfilename_ints)
    intlabels -= np.min(intlabels)
    rgblabels = img_as_ubyte(np.load(directory_output_masks + myfilename_rgb))

    logging.info(f'{TAG}: begin plotting')
    
    ensure_path_exists(stage5_output_dir)

    filename = f'{input_img_basename}_viz_int.png'
    filepath = os.path.join(stage5_output_dir,filename)
    plt.imshow(intlabels)
    plt.savefig(filepath)

    filename = f'{input_img_basename}_viz_rgb.png'
    filepath = os.path.join(stage5_output_dir,filename)
    plt.imshow(rgblabels)
    plt.savefig(filepath)

    logging.info(f'{TAG}: done plotting.')
    logging.info(f'{TAG}: unique int labels: {np.unique(intlabels)}')

    int_to_rgb_dict = uf.intlabels_and_rgblabels_to_dict(intlabels, rgblabels, list_to_ignore = [], divide_by_255=True)
    logging.info(f'{TAG}: dtype {intlabels.dtype}. int_to_rgb_dict: {int_to_rgb_dict}.')
    
    int_to_rgb_dict_no_zero = uf.intlabels_and_rgblabels_to_dict(intlabels, rgblabels, list_to_ignore = [0], divide_by_255=True)
    
    return original_image, intlabels, rgblabels, int_to_rgb_dict, int_to_rgb_dict_no_zero


def load_napari(input_file_name):
    TAG = 'load_napari:'
    Image.MAX_IMAGE_PIXELS = None

    logging.info(f'{TAG} begin. napari version = {napari.__version__}')

    viewer = napari.Viewer()

    colors = load_viz_colors()
    original, intlabels, rgblabels, int_to_rgb_dict, int_to_rgb_dict_no_zero = viz_load_single_data(input_file_name)

    original_pyr = uf.get_pyramid_hybrid_loading(original, 1000, 200000)
    logging.info(f'{TAG} original pyr shape: {original_pyr}')
    logging.info(f'{TAG} pyramid shapes: {[each.shape for each in original_pyr]}')

    logging.info(f'{TAG} adding original image')
    viewer.add_image(original_pyr, name='Original image')

    #todo: what is this?
    # if 'gt' in globals():
    #     gt_pyr = uf.get_pyramid_hybrid_loading(gt, 1000, 200000)
    #     viewer.add_labels(gt_pyr, name='Groundtruth', visible=False, color=int_to_rgb_dict,)
    # 
    # if 'gt_45' in globals() and True:
    #     gt_45_upsize = uf.upsize(gt_45, 45)
    #     gt_45_pyr = uf.get_pyramid_hybrid_loading(gt_45_upsize, 1000, 200000)
    #     viewer.add_labels(gt_45_pyr, name='Groundtruth_clusters', visible=False, color=df_colors_out)
    #   
    #     output_edges_gt_45 = nf.get_edges_of_cluster_shapes_with_partial_upscale(gt_45, rgb_gt_45, k=6,
    #                                                                     partial_upscale = 5,
    #                                                                     final_upscale = 45,
    #                                                                     image_edges_drawn = True,
    #                                                                     output_type='int')
    #
    #     pyr_cluster_edges_gt_45 = uf.get_pyramid_hybrid_loading(output_edges_gt_45, 1000, 200000)
    #
    #     viewer.add_labels(pyr_cluster_edges_gt_45, color = df_colors_out_no_zero, name='Groundtruth_edges', opacity=1)
    
    partial_upscale = env.partial_upscale if env.partial_upscale else 10
    patchsize = env.patchsize if env.patchsize else 100
    final_target_size = env.final_target_size if env.final_target_size else 1

    #this the partial upscale needed for creating the edges labels layer
    #the higher this number, the smoother the edges are displayed but the slower it takes
    final_upscale = int(patchsize / (final_target_size))
    
    logging.info(f'{TAG} part 1')

    #This next part grays out the areas that are not part of clusters
    if True:
        not_clusters_yet = nf.upsize(intlabels <= 0, patchsize)
        pyr_not_clusters = uf.get_pyramid_hybrid_loading(not_clusters_yet, 1000, 2000000)
        viewer.add_labels(pyr_not_clusters, color=colors.dict_gray_out_colors, 
                        name='Part of any cluster', visible=False)

    logging.info(f'{TAG} part 2')

    #This next part creates a labels layer that shows the edges of the clusters
    if True:
    #    rgbalabels = nf.convert_rgb_to_rgba(rgblabels, transparent=np.array([255, 255, 255]), transparency_val = 255)
        output_edges = nf.get_edges_of_cluster_shapes_with_partial_upscale(intlabels, rgblabels, k=4,
                                                                        partial_upscale = partial_upscale,
                                                                        final_upscale = final_upscale,
                                                                        image_edges_drawn = True,
                                                                        output_type='int')
    #    intlabels_partial_upsize = nf.upsize(intlabels, partial_upscale)
    #    rgblabels_partial_upsize = nf.upsize2(rgblabels, partial_upscale)

    #    output_edges = nf.get_edges_of_cluster_shapes(intlabels_partial_upsize, rgblabels_partial_upsize, k=4)
    #    output_edges_labels_for_int_layer = rgb2labelint(output_edges, rgb_colors_with_inital_grey)    
    #    output_edges_labels_for_int_layer_fullsize = nf.upsize(output_edges_labels_for_int_layer, further_upscale)

    #    pyr_cluster_edges = uf.get_pyramid_hybrid_loading(output_edges_labels_for_int_layer_fullsize, 2000, 2000000)
    #    pyr_cluster_edges = [output_edges[::i, ::i] for i in [1, 2, 4, 8, 16, 32]]
        pyr_cluster_edges = uf.get_pyramid_hybrid_loading(output_edges, 1000, 200000)
    #    viewer.add_labels(pyr_cluster_edges, color=dict_colors_initial_transp_0to1, 
    #                      name='Cluster edges', visible=False, opacity=1)
        viewer.add_labels(pyr_cluster_edges, color = int_to_rgb_dict_no_zero, name='Our edges', opacity=1)

    logging.info(f'{TAG} part 3')

    #This next part shows the clusters as shaded patches
    if True:
        intlabels_upsized = nf.upsize(intlabels, final_upscale)
        intlabels_pyr = uf.get_pyramid_hybrid_loading(intlabels_upsized, 1000, 200000)
        viewer.add_labels(intlabels_pyr, color=int_to_rgb_dict, 
                        name='Our clusters', opacity=0.2, visible=False)
    #    cluster_patches_int = rgb2labelint(rgblabels_partial_upsize, rgb_colors_with_inital_grey)
    #    cluster_patches_int_fullsize = nf.upsize(cluster_patches_int, further_upscale)
    #    img_clusters = uf.get_pyramid_hybrid_loading(cluster_patches_int_fullsize, 2000, 2000000)
    #    viewer.add_labels(img_clusters, color=dict_colors_initial_transp_0to1, 
    #                      name='Shaded clusters', opacity=0.2)

    logging.info(f'{TAG} done.')
