import numpy as np
from PIL import Image
from skimage.io import imread

from fastlbp_baseline_imbg import run, Environment

params = Environment()
params.nradii=4
params.patchsize=50
params.nthreads=8
params.stages=[1,2]
params.imgname="img_L_400x400.jpg"

params.indir = "/app/src/data/in"
params.outdir = "/app/src/data/out"

run(params)
