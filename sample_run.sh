# An example script to run the pipeline - do not run
exit 1

# do not forget about your env. You can use venv instead of conda
python3 -m venv env
source env/bin/activate
pip install -r pip_reqs_aln.txt # pip reqs file based on environment_lbp3d_aln.yml

# place input images directly in ./data/in
# output will be in ./data/out/{stage_name}/{image_name}/

# to run stage 1 only
python entrypoint.py --stages=1

# to run stages 2,3,4. They can handle a single image only for now, so --imgname is required
python entrypoint.py --stages=2-4 --imgname=img1.jpg

# to run stages 2,3 with custom input/output dir 
# and custom radii number (e.g. for faster computation)
# and custom threading number (e.g. for 16-core cluster job)
python entrypoint.py --stages=2,3 --imgname=img1.jpg \
    --indir=/path/to/input/dir --outdir=/path/to/output/dir \
    --nradii=4 --nthreads=16

