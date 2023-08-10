
import os
import getopt
from collections import namedtuple
from common import *
import logging

import os
import getopt
from collections import namedtuple
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
from matplotlib.colors import to_rgba

import hdbscan
import umap
import skimage

import useful_functions as uf 

pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 100)

#
# ================= STAGE 3 - UMAP single image ====================
#

def stage3_umap_single(input_file_name):
    input_img_fullpath = os.path.join(env.indir, input_file_name)
    input_img_basename = safe_basename(input_img_fullpath)
    if not os.path.exists(input_img_fullpath):
        logging.error(f'stage3_umap_single: Cannot find any original images in \'{env.indir}\' that look like {input_file_name}')
        sys.exit(1)

    logging.info(f'stage3_umap_single: begin. input_img_basename={input_img_basename}')

    stage2_output_dir = get_outdir(STAGE2, input_img_basename)
    stage3_output_dir = get_outdir(STAGE3, input_img_basename)

    this_shape = get_dims_from_image(input_img_fullpath)
    image_width = this_shape[0]
    image_height = this_shape[1]

    mycolors = uf.return_color_scale('block_colors_for_labels_against_white_small_points')

    directory_X_scaled = stage2_output_dir
    filename_X_scaled = f'{input_img_basename}_LBP_X.npy'
    X = np.load(os.path.join(directory_X_scaled, filename_X_scaled), mmap_mode='r')
    logging.info(f'stage3_umap_single: Loading {filename_X_scaled}: {X.shape}, {X.dtype}')

    start = datetime.now(); print(start)
    directory_OBS = directory_X_scaled
    filename_OBS = f'{input_img_basename}_LBP_OBS.csv'
    df_OBS = pd.read_csv(os.path.join(directory_OBS, filename_OBS), index_col=0)
    df_OBS['Groundtruth'] = pd.Categorical(df_OBS['Groundtruth'])
    df_OBS['original_index'] = pd.Categorical(df_OBS['original_index'])
    logging.info(f'stage3_umap_single: OBS loading took {datetime.now()-start}, shape {df_OBS.shape}')
    logging.info(df_OBS.head())

    directory_VAR = directory_X_scaled
    filename_VAR = f'{input_img_basename}_LBP_VAR.csv'
    df_VAR = pd.read_csv(os.path.join(directory_VAR, filename_VAR), index_col=0)
    df_VAR.index = df_VAR.index.astype(str)
    logging.info(f'stage3_umap_single: done VAR loading, shape {df_VAR.shape}')
    logging.info(df_VAR.head())

    dict_colors = {0:to_rgba(mycolors[0]), 
               1:to_rgba(mycolors[1]), 
               2:to_rgba(mycolors[2]),
               3:to_rgba(mycolors[3]),
               4:to_rgba(mycolors[4]),
               5:to_rgba(mycolors[5]),
               6:to_rgba(mycolors[6]),
               7:to_rgba(mycolors[7]),
               8:to_rgba(mycolors[8]),
               9:to_rgba(mycolors[9]),
              np.nan:(1, 1, 1)
              }

    dict_color_names = {0:mycolors[0], 
                1:mycolors[1], 
                2:mycolors[2],
                3:mycolors[3],
                4:mycolors[4],
                    5:mycolors[5],
                6:mycolors[6],
                    7:mycolors[7],
                8:mycolors[8],
                    9:mycolors[9],  
                np.nan:(1, 1, 1)
                    }

    logging.info('stage3_umap_single: Begin plotting')

    img_original_index = 0
    list_of_index_img0 = df_OBS.index[(df_OBS['original_index'].isin([img_original_index])) &
                                (~df_OBS['Groundtruth'].isin([]))]
    print(len(list_of_index_img0))
    this_df = df_OBS.loc[list_of_index_img0]
    this_df.index = this_df.index.astype(str)
    anndata_concat = ad.AnnData(X=X[list_of_index_img0], obs=this_df, var=df_VAR, dtype=np.float32)
    Groundtruth_vc = anndata_concat.obs['Groundtruth'].value_counts()
    anndata_concat.obs['Groundtruth'] = anndata_concat.obs['Groundtruth'].cat.set_categories(list(Groundtruth_vc.index[Groundtruth_vc > 0]))
    sc.pp.scale(anndata_concat)
    
    z = [dict_colors[each] for each in list(anndata_concat.obs['Groundtruth'].value_counts().index)]
    fig, ax = plt.subplots(1,3, figsize=(17,4))
    Groundtruth_vc.plot(kind='pie', colors=z, ax=ax[0])
    sc.tl.pca(anndata_concat, svd_solver='auto')
    sc.pl.pca(anndata_concat, color='Groundtruth',
                size=1, palette=dict_colors, components=['1,2'], ax=ax[1], show=False, title='')
    sc.pp.neighbors(anndata_concat, n_neighbors=5, n_pcs=None)
    sc.tl.umap(anndata_concat) #, min_dist=0.0
    sc.pl.umap(anndata_concat, color=['Groundtruth'],          
        palette=dict_colors, show=False, alpha=1, ax=ax[2], title='')
    plt.suptitle('Image original_index = ' + str(img_original_index))
    plt.subplots_adjust(wspace=0.3)
    
    ensure_path_exists(stage3_output_dir)
    
    output_file_name = f'{input_img_basename}_UMAP.png'
    output_file_path = os.path.join(stage3_output_dir, output_file_name)

    logging.info(f'stage3_umap_single: Begin savefig to {output_file_path}')
    plt.savefig(output_file_path)
    logging.info('stage3_umap_single: done')


#
# ================= STAGE 4 - UMAP and clustering ====================
#


def stage4_umap_clustering(input_file_name):
    input_img_fullpath = os.path.join(env.indir, input_file_name)
    input_img_basename = safe_basename(input_img_fullpath)
    if not os.path.exists(input_img_fullpath):
        logging.error(f'stage4_umap_clustering: Cannot find any original images in \'{env.indir}\' that look like {input_file_name}')
        sys.exit(1)
        
    logging.info(f'stage4_umap_clustering: begin. input_img_basename={input_img_basename}')

    stage2_output_dir = get_outdir(STAGE2, input_img_basename)
    stage4_output_dir = get_outdir(STAGE4, input_img_basename)

    this_shape = get_dims_from_image(input_img_fullpath)
    image_width = this_shape[0]
    image_height = this_shape[1]

    mycolors = uf.return_color_scale('block_colors_for_labels_against_white_small_points')
    gray = (0.6, 0.6, 0.6)
    dict_colors = {-1:to_rgba(gray),
               0:to_rgba(mycolors[0]), 
               1:to_rgba(mycolors[1]), 
               2:to_rgba(mycolors[2]),
               3:to_rgba(mycolors[3]),
               4:to_rgba(mycolors[4]),
               5:to_rgba(mycolors[5]),
               6:to_rgba(mycolors[6]),
               7:to_rgba(mycolors[7]),
               8:to_rgba(mycolors[8]),
               9:to_rgba(mycolors[9]),
              np.nan:(1, 1, 1)
              }
    dict_color_names = {0:mycolors[0], 
                1:mycolors[1], 
                2:mycolors[2],
                3:mycolors[3],
                4:mycolors[4],
                    5:mycolors[5],
                6:mycolors[6],
                    7:mycolors[7],
                8:mycolors[8],
                    9:mycolors[9],  
                np.nan:(1, 1, 1)
                    }
    
    # note that we use stage 2 output
    directory_X_scaled = stage2_output_dir
    filename_X_scaled = f'{input_img_basename}_LBP_X.npy'
    X = np.load(os.path.join(directory_X_scaled, filename_X_scaled), mmap_mode='r')
    logging.info(f'stage4_umap_clustering: Loading {filename_X_scaled}: {X.shape}, {X.dtype}')

    start = datetime.now(); print(start)
    directory_OBS = directory_X_scaled
    filename_OBS = f'{input_img_basename}_LBP_OBS.csv'
    df_OBS = pd.read_csv(os.path.join(directory_OBS, filename_OBS), index_col=0)
    df_OBS['Groundtruth'] = pd.Categorical(df_OBS['Groundtruth'])
    df_OBS['original_index'] = pd.Categorical(df_OBS['original_index'])
    logging.info(f'stage4_umap_clustering: OBS loading took {datetime.now()-start}, shape {df_OBS.shape}')
    logging.info(df_OBS.head())

    directory_VAR = directory_X_scaled
    filename_VAR = f'{input_img_basename}_LBP_VAR.csv'
    df_VAR = pd.read_csv(os.path.join(directory_VAR, filename_VAR), index_col=0)
    df_VAR.index = df_VAR.index.astype(str)
    logging.info(f'stage4_umap_clustering: done VAR loading, shape {df_VAR.shape}')
    logging.info(df_VAR.head())


    C, this_image_index = 0, 0
    
    # =========
    list_of_index_img0 = df_OBS.index[(df_OBS['original_index'].isin([this_image_index])) &
                                     (~df_OBS['Groundtruth'].isin([]))] #can add zero in here to remove the background class
    print(this_image_index)
    y = X[list_of_index_img0.astype(int)]
    anndata_concat = ad.AnnData(X=y, obs=df_OBS.loc[list_of_index_img0], var=df_VAR, dtype=np.float32)
    mycolors_list_all = anndata_concat.obs['Groundtruth'].map(dict_color_names)
    mycolors_list_all
    
    Groundtruth_vc = anndata_concat.obs['Groundtruth'].value_counts()
    anndata_concat.obs['Groundtruth'] = anndata_concat.obs['Groundtruth'].cat.set_categories(list(Groundtruth_vc.index[Groundtruth_vc > 0]))
    sc.pp.scale(anndata_concat)
    z = [dict_colors[each] for each in list(anndata_concat.obs['Groundtruth'].value_counts().index)]
    sc.tl.pca(anndata_concat, svd_solver='auto')
    sc.pp.neighbors(anndata_concat, n_neighbors=15, n_pcs=None)
    sc.tl.umap(anndata_concat) #, min_dist=0.0
    
    clusterable_embedding2 = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=20,
        random_state=42,
    ).fit_transform(anndata_concat.X)

    clusterer2 = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=1, cluster_selection_epsilon=0.0,gen_min_span_tree=True)
    clusterer2.fit(clusterable_embedding2)
    anndata_concat.obs['HDBScan_UMAP_clusters'] = pd.Categorical(clusterer2.labels_)#.map({-1:-1, 0:3, 1:9})

#    original_fname_annotated = pd.unique(anndata_concat.obs['output_fname_annotated'])[0]
#    annotated = np.load(directory_annotated + original_fname_annotated.replace('.jpg', '.npy') )
#    annotated_recolor = uf.label2rgb_with_dict(annotated, dict_color_names)
#    annotated_rescaled = np.load(directory_annotated_rescaled + original_fname_annotated.replace('.jpg', '.npy') )

    dims = (np.max(anndata_concat.obs['X0']) + 1, np.max(anndata_concat.obs['X1']) + 1)
    heatmap = uf.create_heatmap_from_list_of_coords(dims, anndata_concat.obs['X0'], anndata_concat.obs['X1'], 
                                           anndata_concat.obs['HDBScan_UMAP_clusters'], scale_up = 1,
                                           bg_value=-2, dtype=np.uint8)
#    this_colors_dict = {key:value for key, value in zip(anndata_concat.obs['HDBScan_UMAP_clusters'].cat.categories,
#                                                                                        anndata_concat.uns['HDBScan_UMAP_clusters_colors'])}
#    this_colors_dict = dict_colors
#    this_colors_dict[-2] = mycolors[0]
#    heatmap_recolor = uf.label2rgb_with_dict(heatmap, this_colors_dict)
#    annotated_rescaled = annotated_rescaled[0:heatmap.shape[0], 0:heatmap.shape[1]]
#    classes_mapping_dict, overlap = uf.get_optimal_overlap_of_classes(heatmap, annotated_rescaled, 
#                                                                  elements_to_remove_from_observed = [-2, -1])
    # -2 is the class that is background I think
#    observed_classes_to_colors = {key: dict_colors[value] for key,value in classes_mapping_dict.items() if value is not np.nan}
#    observed_classes_to_colors[-2] = (1.0, 1.0, 1.0, 1.0)
#    observed_classes_to_colors[-1] = (0.3, 0.3, 0.3, 1.0)
#    remaining_classes = list(set(classes_mapping_dict.keys()) - set(observed_classes_to_colors.keys()))
#    N_rem = len(remaining_classes)
#    for i, each in enumerate(remaining_classes):
    #    observed_classes_to_colors[each] = ((i+1)/(N_rem+1), (i+1)/(N_rem+1), (i+1)/(N_rem+1), 1.0)
#        observed_classes_to_colors[each] = to_rgba(colors_new[i])
#    heatmap_recolor = uf.label2rgb_with_dict(heatmap, observed_classes_to_colors)
    heatmap_recolor = uf.label2rgb_with_dict(heatmap, dict_colors)
    
#    observed_classes_to_colors_for_umap = {key:to_hex(value) for key, value in observed_classes_to_colors.items()
#                                          if key in heatmap}
#    sorted_keys = sorted(observed_classes_to_colors_for_umap.keys())
#    sorted_keys.remove(-2)
#    anndata_concat.uns['HDBScan_UMAP_clusters_colors'] = [observed_classes_to_colors_for_umap[each] for
#                                                          each in sorted_keys]
    
    
    logging.info(f'stage4_umap_clustering: Begin plotting')

    gridspec = dict(wspace=0.0, width_ratios=[1, 0.1, 1, 0.1, 1, 0.25, 1, 0.4, 1])
    fig, ax = plt.subplots(1,9, figsize=(18,4), gridspec_kw=gridspec) #, 
#    sc.pl.pca(anndata_concat, color='Groundtruth',
#                 size=1, palette=dict_colors, components=['1,2'], ax=ax[0], show=False,
#             title='PCA with colors\nfrom groundtruth')
    
    # img_name = pd.unique(anndata_concat.obs['output_filename'])[0]
    img = skimage.io.imread(input_img_fullpath) # was directory_images + img_name
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original image')
    sc.pl.umap(anndata_concat, color=['Groundtruth'],          
                 palette=dict_colors, show=False, alpha=1, ax=ax[6],
              title='UMAP with colors from \n repository groundtruth')
    sc.pl.umap(anndata_concat, color=['HDBScan_UMAP_clusters'], 
                 size=20, show=False, alpha=0.8, ax=ax[8],
              title='UMAP with colors\nfrom my clusters')
#    ax[2].imshow(annotated_recolor, interpolation='none')
    ax[2].axis('off')
    ax[2].set_title('Groundtruth image from\nrepository annotations')
#    ax[4].imshow(heatmap_recolor, interpolation='none')
    ax[4].imshow(heatmap, interpolation='none')
    ax[4].axis('off')
    ax[4].set_title('Image of my clusters')
    for this_ax in [ax[1], ax[3], ax[5], ax[7]]:
        this_ax.axis('off')
    plt.suptitle('image_index = ' + str(this_image_index), y=1.03)
    # plt.savefig('image_index_' + str(this_image_index) + '.png', dpi=400, bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    
    ensure_path_exists(stage4_output_dir)
    
    output_file_name = f'{input_img_basename}_Clusters.png'
    output_file_path = os.path.join(stage4_output_dir, output_file_name)

    logging.info(f'stage4_umap_clustering: Begin savefig to {output_file_path}')
    plt.savefig(output_file_path)
    
    ints_filename = f'{input_img_basename}_Clusters_ints.npy'
    ints_path = os.path.join(stage4_output_dir, ints_filename)
    logging.info(f'stage4_umap_clustering: Begin np.save heatmap/ints to {ints_path}')
    np.save(ints_path, heatmap)
    
    rgb_filename = f'{input_img_basename}_Clusters_rgb.npy'
    rgb_path = os.path.join(stage4_output_dir, rgb_filename)
    logging.info(f'stage4_umap_clustering: Begin np.save heatmap_recolor/rgb to {rgb_path}')
    np.save(rgb_path, heatmap_recolor)


    logging.info('stage4_umap_clustering: Done')
    
