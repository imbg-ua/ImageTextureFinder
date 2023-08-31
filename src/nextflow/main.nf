#!/usr/bin/env/ nextflow

//
// Based on https://github.com/BioinfoTongLI/hcs_analysis/blob/main/main.nf
//

nextflow.enable.dsl=2

// you should set these as `nextflow run` params
params.input_dir = 'host/data/in'   // intentionally malformed
params.output_dir = 'host/data/out' // intentionally malformed


// list of images to process
params.images = [
    [ "id": "img1.jpg", "nradii": 2 ]
]

docker_params = '--gpus all -v /lustre:/lustre -v /nfs:/nfs'
docker_params = "${docker_params} -v ${params.input_dir}:/data/in -v ${params.output_dir}:/data/out"

process fastlbp_alpha {
    debug true
    cache true

    container 'mkrooted/imbg-fastlbp:latest'
    containerOptions "${workflow.containerEngine == 'singularity' ? '-B /lustre,/nfs --nv' : docker_params}"
    publishDir params.output_dir, mode:"copy"

    input:
      val(meta)

    output:
      val(meta)

    // CHECK fastlbp PARAMS BEFORE RUNNING NF PROCESS
    script:
      def args = task.ext.args ?: ''
      """
      mkdir -p /data/out/logs
      python -m fastlbp_imbg \
          --stages=2 \
          --indir=/data/in \
          --outdir=/data/out \
          --nradii=${meta['nradii']} \
          --imgname=${meta['id']} \
          ${args} > /data/out/logs/nf_${task.index}.log
      """
}

jobs = channel.from(params.images)

workflow  {
    fastlbp_alpha(jobs)
}
