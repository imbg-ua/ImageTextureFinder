# ImageTextureFinder
A project to create an easy-to-use way of finding areas of common patterns and structures within an image. Should work on any image, designed for use on any biological images including DAPI, IMC and H&E.

See `sample_run.sh` for details.

- Branch `pip` is the latest. It contains a pip packagable version of this project.
- Branch `baseline` is the most stable. It is not ready for pip tho.
- Branch `dev` is unstable and for dev purpuses only.


## Container
Image tag is `mkrooted/imbg-fastlbp`. Hosted on Docker Hub (https://hub.docker.com/repository/docker/mkrooted/imbg-fastlbp/general).
See [Dockerfile](src/container/Dockerfile) for details.

## Nextflow
The idea is to use `mkrooted/imbg-fastlbp` as execution environment. 
Note that Singularity should be used instead of default Docker. Nextflow can convert Docker container to a Singularity one automatically.

<!-- TODO: -->
Execution example:
```bash
# working dir is ./src/nextflow
nextflow run -profile local \ 
    --input_dir /mnt/y/IMBG/IMBGImageTextureFinder/data/in \
    --output_dir /mnt/y/IMBG/IMBGImageTextureFinder/data/out \
    .
```

See [main.nf](src/nextflow/main.nf) for details. Unfinished though.


---

# Guides

## How to build and deploy a pip package

Src: https://packaging.python.org/en/latest/tutorials/packaging-projects/

- Add your access token to `.pypirc`
    ```
    # ~/.pypirc 
    [pypi]
      username = __token__
      password = pypi-TOKEN_FROM_YOUR_PYPI_SETTINGS_GOES_HERE
    ```
- Install prerequisites
    ```
    pip install --upgrade twine build
    ```
- Edit project version in `pyproject.toml`
- Build and upload the project
    ```
    cd src/fastlbp_imbg
    python -m build      # .whl and .gz output will be at ./dist directory
    python3 -m twine upload dist/*   # note that this can accidentally upload unneeded builds
    ```

## How to build a docker container
```
cd src/container
docker build -t mkrooted/imbg-fastlbp .   # note the dot
# this can take a lot of time on local pc
```

## How to run fastlbp_imbg inside docker manually
```
docker run -v /full/path/to/host/data/dir:data mkrooted/imbg-fastlbp \
    python -m fastlbp_imbg \
    # --stages=1 --imgname=img1.jpg  \
    # etc etc other parameters go here
```


- -v stands for volume attachment. it gives the container an access to the files on your host OS.  
    Syntax: `-v host_path:container_path`
- `mkrooted/imbg-fastlbp` is image name. should be the same as in `-t` of a `docker build`
- everything after the first line is a command to run inside the container 

