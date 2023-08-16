"""
Refactored by mkrooted256
"""

import getopt
import logging
import sys

from .common import *
from .LBP import *
from .Embedding import *
from .Viz import *

"""
features todo:
- implement a signal handler (sigterm/sigkill) for the parallel computation stage
"""

#
# ================= SETTING UP ====================
#

logging.basicConfig(encoding='utf-8', level=logging.INFO)


def parse_args():
    """
    todo: 
    - validate paths if we are on a cluster; enabled by default and an explicit cmd flag to disable?
    """
    logging.info('args:')
    logging.info(sys.argv)
    
    # Parse cli arguments
    optlist, args = getopt.getopt(sys.argv[1:], "", ['stages=', 'indir=', 'outdir=', 'nthreads=', 'nradii=', 'patchsize=', 'imgname='])
    indir_set, outdir_set = False, False

    for opt,val in optlist:
        if opt == '--indir': 
            indir_set = True
            env.indir = val
        if opt == '--outdir': 
            outdir_set = True
            env.outdir = val
        if opt == '--imgname':
            env.imgname = val
        if opt == '--nthreads':
            env.nthreads = int(val)
        if opt == '--nradii':
            env.nradii = int(val)
        if opt == '--patchsize':
            env.patchsize = int(val)
            
        if opt == '--stages':
            # stages in a format of `1,2,3,4,5` or `1-5` (incl.)
            if '-' in val and ',' in val:
                logging.error("parse_args: Mixed , and - usage in `stages` parameter. I am not going to parse this.")
                return False
            if ',' in val:
                env.stages = list(map(int,val.split(',')))
                logging.info('parse_args: parsing stages as list')
            elif '-' in val:
                nums = list(map(int, val.split('-')))
                env.stages = list(range(nums[0],nums[1]+1))
                logging.info('parse_args: parsing stages as range')
            else:
                nums = [int(val)]
                env.stages = nums
                logging.info('parse_args: parsing stages as a single num')
    # end for
    
    logging.info('optlist:')
    logging.info(optlist)
    
    logging.info('Environment:')
    logging.info(env)

    if not indir_set:
        logging.warning(f"setup_environment: `--indir=` arg not set. using default {env.indir}")
    if not outdir_set:
        logging.warning(f"setup_environment: `--outdir=` arg not set. using default {env.outdir}")

    return True

def prerequisites():
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', None)
    Image.MAX_IMAGE_PIXELS = None # to skip compression bomb check in PIL

    logging.info(f'prerequisites: Got stages {env.stages}')
    if not env.stages:
        logging.error('prerequisites: No stages to execute. `--stages=` argument is required. Aborting')
        return False

    if 2 in env.stages and not env.imgname:
        logging.error('prerequisites: Stage 2 requested but no imgname provided. `--imgname=` parameter is required')
        return False
    
    return True

#
# ================= MAIN ====================
#

def run_stages():
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

    if 5 in env.stages:
        logging.info('STAGE 5 BEGIN')
        start = datetime.now();
        load_napari(env.imgname)
        logging.info(f'STAGE 5 END. took {datetime.now()-start}')



def main():
    logging.info('Henlo')

    if not parse_args():
        logging.error('Arg parsing failed')
        sys.exit(1)
    if not prerequisites():
        logging.error('Prereqs failed')
        sys.exit(1)
    
    logging.info('Env ok')
    run_stages()
    logging.info('Goodbye')
    
    return


"""
An alternative entrypoint to run the pipeline from python (e.g. from jupyter notebook) instead of CLI
"""
def run(environment:Environment):
    logging.info('Henlo')
    logging.info(environment)
    
    global env
    env = environment
    logging.info(environment)
    
    if not prerequisites():
        logging.error('Prereqs failed')
        sys.exit(1)
        
    logging.info('Env ok.')
    run_stages()
    logging.info('Goodbye')
    
    return


if __name__ == "__main__":
    main()
    sys.exit(0)

