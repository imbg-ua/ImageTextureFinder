"""
Refactored by mkrooted256
"""

import os
import getopt
from collections import namedtuple
from utils import ensure_path_exists, get_dims_tiff, get_npoints_for_radius, get_numpy_datatype_unsigned_int, get_radii, get_infile_extention_regex
import logging

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import cv2
import skimage
import tifffile
import os
import pandas as pd
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import cv2
import skimage
import tifffile
import os
import matplotlib
import matplotlib.patheffects as path_effects
from functools import partial
from skimage.feature import local_binary_pattern
from importlib import reload
from itertools import repeat
from datetime import datetime
from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed
from numba import njit

import useful_functions as uf 

"""
features todo:
- implement a signal handler (sigterm/sigkill) for the parallel computation stage
"""

# Current working directory
cwd = os.getcwd();

Environment = namedtuple('Environment', 'indir, outdir, nthreads, nradii')
env = Environment(
    indir=os.path.join(cwd, 'data', 'in'),
    outdir=os.path.join(cwd, 'data', 'out'),
    nthreads=8,
    nradii=15
)

#
# ================= SETTING UP ====================
#

def setup_environment():
    """
    todo: 
    - determine input/output directories;
    - check if they exist and if permissions are ok; create if not;
    - validate paths if we are on a cluster; enabled by default and an explicit cmd flag to disable?
    - number of threads 
    """
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    Image.MAX_IMAGE_PIXELS = None # to skip compression bomb check in PIL

    return True



# Get list of files from indir. Generate a dataframe with a list of jobs
def prepare_jobs():
    radius_list = get_radii(env.nradii)
    channel_list = [0,1,2]

    # Find all input files
    img_list = [] 
    with os.scandir(env.indir) as scandir:
        logging.info("Searching for input files in '{}'.".format(env.indir))
        ext_filter = get_infile_extention_regex()
        for entry in scandir:
            if entry.name.startswith('.') or not entry.is_file():
                continue
            if not ext_filter.match(entry.name):
                logging.warn("File '{}' has an unsupported format".format(entry.name))
                continue
            img_list.append(entry.path)
    logging.info("Input files:\n%s", '\n'.join(img_list))

    # Generate job list
    df_all = pd.DataFrame(img_list, columns=['Filenames'])
    df_all['dims'] = list(map(get_dims_tiff, [env.indir]*len(df_all) + df_all['Filenames']))
    df_all = uf.create_df_cross_for_separate_dfs(df_all,
                                                pd.DataFrame(radius_list, columns=['radius']))
    df_all = uf.create_df_cross_for_separate_dfs(df_all,
                                                pd.DataFrame(channel_list , columns=['channel']))
    df_all['n_points'] = get_npoints_for_radius(df_all['radius']) # todo: check if broadcasting works as intended
    df_all['method'] = 'uniform'

    df_all['safe_infilename'] = df_all['Filenames'].str.replace(pat='.', repl='_', regex=False)
    df_all['Subfolder'] = df_all.apply(lambda row: os.path.join(env.outdir, row['safe_infilename']), axis=0)
    df_all['outfilename'] = df_all.apply(lambda row: '{}_ch{}_r{}.npy'.format(row['safe_infilename'], row['channel'], row['radius']), axis=0)
    df_all['Fpath_out'] = df_all.apply(lambda row: os.path.join(df_all['Subfolder'], df_all['outfilename']), axis=0)
    
    df_all['patchsize'] = 100

    df_all.sort_values(['n_points', 'Filenames'], inplace=True, ascending=True)
    df_all.reset_index(drop=True, inplace=True)

    max_npoints = df_all['n_points'].max()
    mydtype = get_numpy_datatype_unsigned_int(max_npoints);
    df_all['mydtype'] = mydtype
    logging.info('dtype is {}'.format(mydtype))

    # check for existing output directories
    df_all['Subfolder_exists'] = df_all.apply(lambda row: ensure_path_exists(row['Subfolder']))
    logging.info(df_all['Subfolder_exists'].value_counts)

    df_all['Fpath_out_exists'] = df_all.apply(lambda row: os.path.exists(row['Fpath_out']))
    logging.info(df_all['Fpath_out_exists'].value_counts)

    logging.debug('job list:')
    logging.debug(df_all)
    return df_all

#
# ================= COMPUTATION ====================
#

@njit
def bincount_the_patches(lbp, patchsize, n_points):
    sizex0 = int(lbp.shape[0]/patchsize)
    sizex1 = int(lbp.shape[1]/patchsize)
    output = np.zeros((sizex0, sizex1, n_points+2), dtype=lbp.dtype)
    for i in range(sizex0):
        for j in range(sizex1):
            this_patch = lbp[i*patchsize:(i+1)*patchsize, 
                             j*patchsize:(j+1)*patchsize]
            mybincount = np.bincount(this_patch.ravel())
            output[i,j, 0:len(mybincount)] = mybincount
    return output

def apply_LBP_to_img(img, current_jobs, job_idx):
    job = current_jobs.iloc[job_idx]
    lbp = local_binary_pattern(
        img[:,:, job['channel']], 
        job['n_points'], 
        job['radius'], 
        job['method']
    ).astype(job['mydtype'])
    lbp = uf.crop_to_superpixel(lbp, job['patchsize'])
    lbp_bincounts = bincount_the_patches(lbp, job['patchsize'], job['n_points'])
    np.save(job['Fpath_out'], lbp_bincounts)

    return 0

# formerly 'load_images'
def process_single_input_image(image_fname, df_jobs, input_dir):
    logging.info("Begin processing '%s'", image_fname)

    Image.MAX_IMAGE_PIXELS = None
    img = skimage.io.imread(os.path.join(input_dir, image_fname))
    current_jobs = df_jobs.loc[df_jobs['Filenames'] == image_fname]

    Parallel(n_jobs=1)(
        ( delayed(apply_LBP_to_img)(img, current_jobs, i) ) for i in trange(len(current_jobs))
    )

    logging.info("End processing '%s'", image_fname)

def process_all_images(df_pending_jobs, input_dir, n_threads, input_imgs):
    logging.info('Begin running all pending jobs (%d).', df_pending_jobs.shape[0])
    Parallel(n_jobs=n_threads)(
        ( delayed(process_single_input_image)(img_fname, df_pending_jobs, input_dir) )
            for img_fname in tqdm(input_imgs, total=len(input_imgs))
    )
    logging.info('End running all pending jobs.')

#
# ================= MAIN ====================
#


def main():
    optlist, args = getopt.getopt(sys.argv, "", ['indir=', 'outdir=', 'nthreads=', 'nradii=', 'skip-cluster-checks'])
    for opt,val in optlist[0]:
        if opt in ['indir', 'outdir', 'nthreads', 'nradii']:
            env[opt] = val
    reload(uf)
    if not setup_environment():
        logging.error('Env not ok')
        sys.exit(1)

    logging.info('Env ok')

    df_jobs = prepare_jobs()
    df_pending_jobs = df_jobs.loc[df_jobs['Fpath_out_exists'] == False]
    logging.info('pending jobs:\n%s', '\n'.join(df_pending_jobs['outfilename']))

    img_list = list(df_pending_jobs['Filenames'].unique())
    process_all_images(df_pending_jobs, env.indir, env.nthreads, img_list)

    logging.info('Goodbye!')
    return


if __name__ == "__main__":
    main()
    sys.exit(0)

