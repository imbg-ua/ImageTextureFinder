
# from collections import namedtuple
# from namedlist import namedlist
from dataclasses import dataclass
from PIL import Image
import re
import numpy as np
import os

cwd = os.getcwd();

@dataclass
class Environment:
    indir:str     =os.path.join(cwd, 'data', 'in')
    outdir:str    =os.path.join(cwd, 'data', 'out')
    nthreads:int  =8
    nradii:int    =15
    patchsize:int =100
    stages:list   =None
    imgname:str   =None
    final_target_size:int =1
    partial_upscale:int   =10

global env
env = Environment()


COLORS=['#0072b2','#d55e00','#009e73', '#cc79a7','#f0e442','#56b4e9']

OUTPUT_DIRS = [
    '1_lbp_output',
    '2_patches',
    '3_umap',
    '4_clustering', # was 'HDBScan_Output_masks'
    '5_viz'
]

# Stage literals
STAGE1, STAGE2, STAGE3, STAGE4, STAGE5 = 1,2,3,4,5

"""
stage: int \in {1,2,3,4,5}
img_basename: should be a safename of an image (see `safe_basename``)
"""
def get_outdir(stage, img_basename):
    stage = int(stage)
    if not (1 <= stage <= 5): raise ValueError('Stage should be in {1,2,3,4,5}')
    return os.path.join(env.outdir, OUTPUT_DIRS[stage-1], img_basename)

def get_radii(n=15):
    radius_list = [round(1.499*1.327**(float(x))) for x in range(0, n)]
    return radius_list

def get_npoints_for_radius(r):
    return np.ceil(2*np.pi*r)

# default color palette of 6 colors
def get_colors():
    return ['#0072b2','#009e73','#d55e00', '#cc79a7','#f0e442','#56b4e9']

def get_dims_from_image(filepath):
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filepath, mode='r') # PIL.Image.open loads only file header, not the actual raster data 
    channels_num = len(image.getbands())
    return (image.width, image.height, channels_num) # todo: i bloody hope the order is correct

# regex to filter supported input file extentions
def get_infile_extention_regex():
    return re.compile(r"^.*\.(tif|tiff|jpg|jpeg|png)$", re.IGNORECASE)

# todo: wtf is this
def get_numpy_datatype_unsigned_int(largest_value):
    if largest_value <= 255:
        print('In lower 255')
        value = np.uint8
    elif largest_value <= 65535:
        value = np.uint16
    else:
        value = 0
    return value

def ensure_path_exists(path):
    os.makedirs(path, exist_ok=True)
    return os.path.exists(path)

# return file name and ext and remove all dangerous chars like '.'
def safe_basename(path):
    return os.path.basename(path).replace('.','_')

@dataclass
class Stage1Method:
    basename:str
    channel:int
    radius:int
    npoints:int

def generate_stage1_filename(basename, channel, radius, npoints):
    return '{}_lbp_ch{}_r{}_n{}.npy'.format(basename, int(channel), int(radius), int(npoints))

def parse_stage1_filename(name):
    tok = name.split('.')[0] # remove all extentions.
    tok = name.rsplit('_', 3) # at most 3 separators. group leftmost ones
    basename, channel, radius, npoints = tok # explicit to raise error if smthng is wrong
    return Stage1Method(basename, int(channel[2:]), int(radius[1:]), int(npoints[1:]))
