#!/usr/bin/env/ nextflow

//
// Based on https://github.com/BioinfoTongLI/hcs_analysis/blob/main/main.nf
//

nextflow.enable.dsl=2
OUTPUT_DIR = 'data/out'
params.images = [
    [ "id": "img1.jpg", "nradii": 4 ]
]

process fastlbp_alpha {
    debug true
    cache true

    container 'mkrooted/imbg-fastlbp:latest'
    containerOptions "${workflow.containerEngine == 'singularity' ? '-B /lustre,/nfs --nv':'-v /lustre:/lustre -v /nfs:/nfs --gpus all'}"
    publishDir OUTPUT_DIR, mode:"copy"

    input:
    val(meta)

    output:
    val(meta)

    script:
    def args = task.ext.args ?: ''
    """
    python -m fastlbp_imbg \
        --stages=1,2,4 \
        --indir=data/in \
        --outdir=${OUTPUT_DIR} \
        --nradii=${meta['nradii']} \
        --imgname=${meta['id']}
        ${args}
    """
}

jobs = channel.from(params.images)

workflow  {
    fastlbp_alpha(jobs)
}
