
from collections import namedtuple
import skimage
from PIL import Image
import re
import numpy as np
import os

def get_radii(n=15):
    radius_list = [round(1.499*1.327**(float(x))) for x in range(0, n)]
    return radius_list

def get_npoints_for_radius(r):
    return np.ceil(2*np.pi*r)

# default color palette of 6 colors
def get_colors():
    return ['#0072b2','#009e73','#d55e00', '#cc79a7','#f0e442','#56b4e9']

def get_dims_tiff(filepath):
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filepath, mode='r') # PIL.Image.open loads only file header, not the actual raster data 
    channels_num = len(image.getbands())
    return (image.height, image.width, channels_num)

# regex to filter supported input file extentions
def get_infile_extention_regex():
    return re.compile(r".(tif|tiff|jpg|jpeg)$", re.IGNORECASE)

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

Stage1Method = namedtuple('Stage1Method', 'basename, channel, radius, npoints')

def generate_stage1_filename(basename, channel, radius, npoints):
    return '{}_lbp_ch{}_r{}_n{}.npy'.format(basename, channel, radius, npoints)

def parse_stage1_filename(name):
    tok = name.split('.')[0] # remove all extentions
    tok = name.split('_')
    basename, channel, radius, npoints = tok # explicit to raise error if smthng is wrong
    return Stage1Method(basename, channel, radius, npoints)
