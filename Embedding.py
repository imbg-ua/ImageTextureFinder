
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
import math
from PIL import Image
import skimage
import os
import pandas as pd
import sys

from skimage.feature import local_binary_pattern
from datetime import datetime
from tqdm import tqdm
from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed
from numba import njit
import anndata as ad

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from tqdm import tqdm, trange
from matplotlib.colors import to_rgba
#import hdbscan
import umap
from tqdm import tqdm, trange
import fastremap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from numba import njit
import scipy
import pickle

import seaborn as sns

import useful_functions as uf 
import numba_funcs as nf

pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 100)

#
# ================= STAGE 3 - UMAP single image ====================
#

def umap_single(input_file_name):
    input_img_fullpath = os.path.join(env.indir, input_file_name)
    input_img_basename = safe_basename(input_img_fullpath)
    if not os.path.exists(input_img_fullpath):
        logging.error(f'umap_single: Cannot find any original images in \'{env.indir}\' that look like {input_file_name}')
        sys.exit(1)

    stage1_output_dir = os.path.join(env.outdir, OUTPUT_DIRS[0])
    stage2_output_dir = os.path.join(env.outdir, OUTPUT_DIRS[1])
    stage3_output_dir = os.path.join(env.outdir, OUTPUT_DIRS[2])

    this_shape = get_dims_tiff(input_img_fullpath)
    image_width = this_shape[0]
    image_height = this_shape[1]

    mycolors = uf.return_color_scale('block_colors_for_labels_against_white_small_points')

    directory_X_scaled = os.path.join(stage2_output_dir, input_img_basename)
    filename_X_scaled = f'{input_img_basename}_LBP_X.npy'
    X = np.load(os.path.join(directory_X_scaled, filename_X_scaled), mmap_mode='r')
    logging.info(f'umap_single: Loading {filename_X_scaled}: {X.shape}, {X.dtype}')

    start = datetime.now(); print(start)
    directory_OBS = directory_X_scaled
    filename_OBS = f'{input_img_basename}_LBP_OBS.csv'
    df_OBS = pd.read_csv(os.path.join(directory_OBS, filename_OBS), index_col=0)
    df_OBS['Groundtruth'] = pd.Categorical(df_OBS['Groundtruth'])
    df_OBS['original_index'] = pd.Categorical(df_OBS['original_index'])
    logging.info(f'umap_single: OBS loading took {datetime.now()-start}, shape {df_OBS.shape}')
    logging.debug(df_OBS.head())

    directory_VAR = directory_X_scaled
    filename_VAR = f'{input_img_basename}_LBP_VAR.csv'
    df_VAR = pd.read_csv(os.path.join(directory_VAR, filename_VAR), index_col=0)
    df_VAR.index = df_VAR.index.astype(str)
    logging.info(f'umap_single: done VAR loading, shape {df_VAR.shape}')
    logging.debug(df_VAR.head())

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

    logging.info('umap_single: Begin plotting')

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
    
    logging.info('umap_single: Begin savefig')
    output_file_name = f'{input_img_basename}_UMAP.png'
    plt.savefig(os.path.join(stage3_output_dir, output_file_name))
    logging.info('umap_single: done')

