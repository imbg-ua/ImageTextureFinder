"""
Refactored by mkrooted256
"""

import os
import getopt
from collections import namedtuple
import logging

import sys
import os

import useful_functions as uf 

from common import *
from LBP import *
from Embedding import *

"""
features todo:
- implement a signal handler (sigterm/sigkill) for the parallel computation stage
"""

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

    # Parse cli arguments
    optlist, args = getopt.getopt(sys.argv, "", ['stages=', 'indir=', 'outdir=', 'nthreads=', 'nradii=', 'patchsize=', 'imgname='])
    indir_set, outdir_set = False, False

    for opt,val in optlist[0]:
        if opt in ['indir', 'outdir', 'imgname']:
            env[opt] = val
            if opt == 'indir': indir_set = True
            if opt == 'outdir': outdir_set = True
        if opt in ['nthreads', 'nradii', 'patchsize']:
            env[opt] = int(val)
        if opt == 'stages':
            # stages in a format of `1,2,3,4,5` or `1-5` (incl.)
            if '-' in val and ',' in val:
                logging.error("setup_environment: Mixed , and - usage in `stages` parameter. I am not going to parse this.")
                return False
            if ',' in val:
                env.stages = map(int,val.split(','))
            elif '-' in val:
                nums = map(int, val.split('-'))
                env.stages = list(range(nums[0],nums[2]+1))
    # end for
    
    if not indir_set:
        logging.warn(f"setup_environment: `--indir=` arg not set. using default {env.indir}")
    if not outdir_set:
        logging.warn(f"setup_environment: `--outdir=` arg not set. using default {env.outdir}")

    if not env.stages:
        logging.error('setup_environment: No stages to execute. `--stages=` argument is required. Aborting')
        return False
    logging.info(f'setup_environment: Got stages {env.stages}')

    if 2 in env.stages and not env.imgname:
        logging.error('Stage 2 requested but no imgname provided. `--imgname=` parameter is required')
        return False
        
    logging.info('Environment:')
    logging.info(env)

    return True

#
# ================= MAIN ====================
#


def main():
    logging.info('Henlo')

    if not setup_environment():
        logging.error('Env not ok')
        sys.exit(1)

    logging.info('Env ok')

    if 1 in env.stages:
        logging.info('STAGE 1 BEGIN')
        start = datetime.now();
        df_jobs = prepare_stage1_jobs()
        df_pending_jobs = df_jobs.loc[df_jobs['Fpath_out_exists'] == False]
        logging.info('pending jobs:\n%s', '\n'.join(df_pending_jobs['outfilename']))

        img_list = list(df_pending_jobs['Filenames'].unique())
        lbp_process_all_images(df_pending_jobs, env.indir, env.nthreads, img_list)
        logging.info(f'STAGE 1 END. took {datetime.now()-start}')

    if 2 in env.stages:
        logging.info('STAGE 2 BEGIN')
        start = datetime.now();
        stage2_single(env.imgname, env.patchsize)
        logging.info(f'STAGE 2 END. took {datetime.now()-start}')

    if 3 in env.stages:
        logging.info('STAGE 3 BEGIN')
        start = datetime.now();
        stage3_umap_single(env.imgname)
        logging.info(f'STAGE 3 END. took {datetime.now()-start}')

    if 4 in env.stages:
        logging.info('STAGE 4 BEGIN')
        start = datetime.now();
        stage4_umap_clustering(env.imgname)
        logging.info(f'STAGE 4 END. took {datetime.now()-start}')


    logging.info('Goodbye')

    return


if __name__ == "__main__":
    main()
    sys.exit(0)

