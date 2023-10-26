
import os
import sys
import logging

import numpy as np
import pandas as pd
import anndata as ad
import math
from PIL import Image
import skimage

from skimage.feature import local_binary_pattern
from datetime import datetime
from tqdm import tqdm
from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed
from numba import njit, jit

from . import useful_functions as uf 
from . import numba_funcs as nf

from .common import get_radii, get_infile_extention_regex, get_dims_from_image, \
    get_npoints_for_radius, OUTPUT_DIRS, generate_stage1_filename, ensure_path_exists, \
    parse_stage1_filename, STAGE1, STAGE2, get_outdir, safe_basename
from . import common

#
# ================= STAGE 1 ====================
#

# Get list of files from indir. Generate a dataframe with a list of jobs
def prepare_stage1_jobs():
    radius_list = get_radii(common.env.nradii)

    # Find all input files
    img_list = [] 
    img_nchannels = []
    with os.scandir(common.env.indir) as scandir:
        logging.info("prepare_stage1_jobs: Searching for input files in '{}'.".format(common.env.indir))
        ext_filter = get_infile_extention_regex()
        for entry in scandir:
            if entry.name.startswith('.') or not entry.is_file():
                continue
            if not ext_filter.match(entry.name):
                logging.warn("prepare_stage1_jobs: File '{}' has an unsupported file extention".format(entry.name))
                continue
            
            try:
                # try to open the image. note: this does not load it in memory, it only reads the file header
                img_desc = Image.open(entry.path)
            except Exception as e:
                logging.warn("prepare_stage1_jobs: Failed to open file '{}' because of the following exception".format(entry.name))
                logging.warn(e, exc_info=True)
                continue
            
            # save number of channels
            img_nchannels.append(len(img_desc.getbands()))
            img_list.append(entry.name)
            
    if not img_list:
        logging.warning("prepare_stage1_jobs: No input files found! Aborting.")
        sys.exit(1)
        
    logging.info("prepare_stage1_jobs: Input files:\n%s", '\n'.join(img_list))

    # Generate job list
    img_cross_channels = []
    for img,nchannels in zip(img_list, img_nchannels):
        for ch in range(nchannels):
            img_cross_channels.append((img, ch))
    df_all = pd.DataFrame(img_cross_channels, columns=['Filenames', 'channel'])
    
    df_all['dims'] = list(map(lambda fname: get_dims_from_image(os.path.join(common.env.indir, fname)), df_all['Filenames']))
    df_all = uf.create_df_cross_for_separate_dfs(df_all,
                                                pd.DataFrame(radius_list, columns=['radius']))

    # df_all = uf.create_df_cross_for_separate_dfs(df_all,
    #                                             pd.DataFrame(channel_list , columns=['channel']))

    df_all['n_points'] = get_npoints_for_radius(df_all['radius']) # todo: check if broadcasting works as intended
    df_all['method'] = 'uniform'

    df_all['safe_infilename'] = df_all['Filenames'].str.replace(pat='.', repl='_', regex=False)
    
    # Output file will be {common.env.outdir}/1_lbp_output/{safe_filename}/{safe_infilename}_ch{channel}_r{radius}.npy
    df_all['Subfolder'] = df_all.apply(lambda row: os.path.join(common.env.outdir, OUTPUT_DIRS[0], row['safe_infilename']), axis=1, result_type='reduce')

    df_all['outfilename'] = df_all.apply(lambda row: generate_stage1_filename(row['safe_infilename'], row['channel'], row['radius'], row['n_points']), axis=1,  result_type='reduce')
    df_all['Fpath_out'] = df_all.apply(lambda row: os.path.join(row['Subfolder'], row['outfilename']), axis=1,  result_type='reduce')
    
    df_all['patchsize'] = 100

    df_all.sort_values(['n_points', 'Filenames'], inplace=True, ascending=True)
    df_all.reset_index(drop=True, inplace=True)

    max_npoints = df_all['n_points'].max()
    mydtype = np.uint16    # get_numpy_datatype_unsigned_int(max_npoints);
    df_all['mydtype'] = mydtype
    logging.info('prepare_stage1_jobs: dtype is {}'.format(mydtype))

    # check for existing output directories
    df_all['Subfolder_exists'] = df_all.apply(lambda row: ensure_path_exists(row['Subfolder']), axis=1, result_type='reduce')
    logging.info(df_all['Subfolder_exists'].value_counts())

    df_all['Fpath_out_exists'] = df_all.apply(lambda row: os.path.exists(row['Fpath_out']), axis=1, result_type='reduce')
    logging.info(df_all['Fpath_out_exists'].value_counts())

    logging.info('job list:')
    logging.info(df_all)
    return df_all

@njit
def bincount_the_patches(lbp, patchsize, n_points):
    sizex0 = int(lbp.shape[0]/patchsize)
    sizex1 = int(lbp.shape[1]/patchsize)
    nfeatures = n_points+2
    output = np.zeros((sizex0, sizex1, nfeatures), dtype=np.uint16) # was dtype=lbp.dtype
    for i in range(sizex0):
        for j in range(sizex1):
            this_patch = lbp[i*patchsize:(i+1)*patchsize, 
                             j*patchsize:(j+1)*patchsize]
            mybincount = np.bincount(this_patch.ravel(), minlength=nfeatures)
            output[i,j,:] = mybincount
    return output

# @jit
def lbp_worker(img_mat, npoints:int, radius:int, method:str, patchsize:int, fpath_out:str):
    # lbp = local_binary_pattern(
    #     img[:,:, job['channel']], 
    #     int(job['n_points']), 
    #     int(job['radius']), 
    #     job['method']
    # ).astype(job['mydtype'])
    # lbp = uf.crop_to_superpixel(lbp, int(job['patchsize']))
    # lbp_bincounts = bincount_the_patches(lbp, int(job['patchsize']), int(job['n_points']))
    # np.save(job['Fpath_out'], lbp_bincounts)
    # np.save(f"{job['Fpath_out']}.imgshp", job['dims']) # save image size as well for futher processing
    
    lbp = local_binary_pattern(
        img_mat, 
        npoints, 
        radius, 
        method
    ).astype(np.uint16)
    lbp = uf.crop_to_superpixel(lbp, patchsize)
    lbp_bincounts = bincount_the_patches(lbp, patchsize, npoints)
    np.save(fpath_out, lbp_bincounts)
    
def apply_LBP_to_img(img, current_jobs, job_idx):
    job = current_jobs.iloc[job_idx]

    lbp_worker(
        img[:,:, job['channel']],
        int(job['n_points']),
        int(job['radius']), 
        job['method'],
        int(job['patchsize']),
        job['Fpath_out']
    )    
    np.save(f"{job['Fpath_out']}.imgshp", job['dims']) # save image size as well for futher processing
    logging.info("apply_LBP_to_img: Done '%s'", job['outfilename'])

# formerly 'load_images'
def lbp_process_single_input_image(image_fname, df_jobs, input_dir):
    logging.info("lbp_process_single_input_image: Begin processing '%s'", image_fname)

    Image.MAX_IMAGE_PIXELS = None
    img = skimage.io.imread(os.path.join(input_dir, image_fname))

    if len(img.shape) != 3:
        logging.info("lbp_process_single_input_image: image has only 2 dimensions, so I will add a new one")
        img = img[:,:,np.newaxis]

    # Select jobs with current filename and 
    nchannels = img.shape[2]
    current_jobs = df_jobs.loc[(df_jobs['Filenames'] == image_fname) & (df_jobs['channel'] < nchannels)]

    Parallel(n_jobs=common.env.nthreads)(
        ( delayed(apply_LBP_to_img)(img, current_jobs, i) ) for i in trange(len(current_jobs))
    )

    logging.info("lbp_process_single_input_image: End processing '%s'", image_fname)

def lbp_process_all_images(df_pending_jobs, input_dir, n_threads, input_imgs):
    logging.info('lbp_process_all_images: Begin running all pending jobs (%d).', df_pending_jobs.shape[0])
    
    # We won't process images in parallel

    # Parallel(n_jobs=n_threads)(
    #     ( delayed(lbp_process_single_input_image)(img_fname, df_pending_jobs, input_dir) )
    #         for img_fname in tqdm(input_imgs, total=len(input_imgs))
    # )

    for i, img_fname in enumerate(input_imgs):
        logging.info(f'lbp_process_all_images: processing image {i+1}/{len(input_imgs)}. fname: {img_fname}')
        lbp_process_single_input_image(img_fname, df_pending_jobs, input_dir)
    
    logging.info('lbp_process_all_images: End running all pending jobs.')

#
# ================= STAGE 2 ====================
#

maxi = 3328 #this is the maximum value that the LBP could get to

# process all results for a single image.
# returns AnnData 
def stage2_worker(input_basename, stage1_output_dir, original_index, output_filename):
    logging.info(f"stage2_worker: begin. {input_basename, stage1_output_dir, original_index, output_filename}")
    
    # find all files of stage1 output dir for a single image.
    # that is, stage1_output_dir should contain .npy files
    data_paths = [] # file paths
    method_names = []  # method names = file basenames
    method_list = []  # List[Stage1Method]
    method_list_cols = []  # List[str]. cartesian product of methods and {0...npoints+1}
    with os.scandir(stage1_output_dir) as sd:
        logging.info(f'stage2_worker: scanning "{stage1_output_dir}" for methods')
        for entry in sd: 
            if entry.is_file and entry.name.endswith('.npy'): 
                name = entry.name[:-4]
                try:
                    method = parse_stage1_filename(name)
                except:
                    continue

                if method.radius > common.env.patchsize:
                    logging.debug(f"stage2_worker: dropping {name} because R > patchsize")
                    continue

                data_paths.append(entry.path)
                method_names.append(name) # filename without extention
                method_list.append(method)
                method_list_cols += [ f"{name}_v{i}" for i in range(0, method.npoints+2) ]
    
    logging.info(f'stage2_worker: found {len(data_paths)} methods. generated {len(method_list_cols)} method_list_cols')
    logging.debug(method_names)
    
    # get shape from any file -- it should be the same across all of them
    fpath_for_shape = data_paths[0]
    this_test_array = np.load(fpath_for_shape, mmap_mode='r')
    this_dtype = this_test_array.dtype 

    # this was floor(shape/patchsize). it is unaltered now because now we calculate histograms in the previous step
    this_shape = this_test_array.shape 


    #this part creates the X0 and X1 coordinates
    x0_array = np.zeros((this_shape[0], this_shape[1]), dtype=np.uint16)
    for i0 in range(this_shape[0]):
        x0_array[i0] = i0
    
    x1_array = np.zeros((this_shape[0], this_shape[1]), dtype=np.uint16)
    for i1 in range(this_shape[1]):
        x1_array[:, i1] = i1
        
    #loading the groundtruth
#    gt = np.load(directory_gt + output_fname_annotated.replace('.jpg', '.npy'))
    
    # method_list_cols is a list of all names of all features for all methods.
    # That is, is has length of sum(n_points_for_ith_method)

    output_array = np.zeros((this_shape[0], this_shape[1], len(method_list_cols)), dtype=this_dtype)
#    print(this_shape, this_dtype, output_array.shape)
    
    # each method has its data in (:,:,(end_of_prev_block+1):(end_of_prev_block+n_features_for_this_method+2))
    start_index = 0
    for idx, (path, method) in enumerate(zip(data_paths, method_list)):
        fpath_to_add = path
        this_npoints = method.npoints
        array_to_add = np.load(fpath_to_add)
        end_index = start_index + this_npoints + 2
        #add to the array
#        print(fpath_to_add, start_index, end_index, array_to_add.shape)
        output_array[:, :, start_index:end_index] = array_to_add
        start_index = end_index


    # That is, now every output_array[x,y,:] is a huge feature vector associated with patch at coords (x,y).
    # Feature vector is a concatenation of all LBP codes for this patch 
    #   from all channels and created using different radius/npoints params.
    
    #this part reshapes the arrays
    reshaped = np.reshape(output_array, (-1, output_array.shape[2]))
    reshaped_x0 = np.reshape(x0_array, (-1))
    reshaped_x1 = np.reshape(x1_array, (-1))
#    reshaped_gt = np.reshape(gt, (-1))
    reshaped_gt = np.zeros(reshaped_x0.shape)
        
    this_AD = ad.AnnData(reshaped, var=method_list_cols, dtype=np.uint16)
    
    this_AD.obs['this_image_index'] = this_AD.obs.index
    this_AD.obs['X0'] = reshaped_x0 
    this_AD.obs['X1'] = reshaped_x1 
    this_AD.obs['Groundtruth'] = reshaped_gt
    this_AD.obs['original_index'] = original_index
    this_AD.obs['output_filename'] = output_filename
#    this_AD.obs['output_fname_annotated'] = output_fname_annotated
    
    logging.info(f"stage2_worker: end. {input_basename, stage1_output_dir, original_index, output_filename}")

    return this_AD

# this is hardcoded to handle the first image only 
def stage2_single(input_file_name, patchsize=100):
    input_img_fullpath = os.path.join(common.env.indir, input_file_name)
    input_img_basename = safe_basename(input_img_fullpath)
    if not os.path.exists(input_img_fullpath):
        logging.error(f'stage2_single: Cannot find any original images in \'{common.env.indir}\' that look like {input_file_name}')
        sys.exit(1)

    stage1_output_dir = get_outdir(STAGE1, input_img_basename)
    stage2_output_dir = get_outdir(STAGE2, input_img_basename)

    this_shape = get_dims_from_image(input_img_fullpath)
    image_width = this_shape[0]
    image_height = this_shape[1]
    
    height_sp = math.floor(image_height/patchsize)
    width_sp = math.floor(image_width/patchsize)
    overall_number_of_patches = height_sp*width_sp

    logging.info(f'stage2: {input_img_basename} ({image_width}*{image_height}): overall_number_of_patches = {height_sp}*{width_sp} = {overall_number_of_patches}')

    anndata_result = stage2_worker(input_file_name, stage1_output_dir, 0, input_img_basename)
    n = anndata_result.X.shape[0]
    out_array_shortened = anndata_result.X
    obs_concat = pd.concat([anndata_result.obs], ignore_index=True)
    logging.info(f'stage2: ad.__version__ = {ad.__version__}')

    ensure_path_exists(stage2_output_dir)
    
    start = datetime.now();
    output_path = os.path.join(stage2_output_dir, f'{input_img_basename}_LBP_X.npy')
    logging.info(f'stage2: saving results to {output_path}...')
    np.save(output_path, out_array_shortened)
    logging.info(f'stage2: done saving results. took {datetime.now()-start}')

    obs_output_path = os.path.join(stage2_output_dir, f'{input_img_basename}_LBP_OBS.csv')
    var_output_path = os.path.join(stage2_output_dir, f'{input_img_basename}_LBP_VAR.csv')
    logging.info(f'stage2: saving metadata to {stage2_output_dir}...')
    obs_concat.to_csv(obs_output_path)
    anndata_result.var.to_csv(var_output_path) # todo: check if `var` is the right field 
    logging.info(f'stage2: done saving metadata. took {datetime.now()-start}')
