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
<!-- Execution example:
```bash
# working dir is ./src/nextflow
nextflow .
``` -->

See [main.nf](src/nextflow/main.nf) for details. Unfinished though.
