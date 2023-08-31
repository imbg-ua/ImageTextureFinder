# based on https://github.com/BioinfoTongLI/hcs_analysis/blob/main/container/Dockerfile
#
FROM  nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04



# Install update apt and install wget package
RUN apt-get update && \
    apt-get install -y wget

# Create user analysis
RUN useradd -m imbg

# switch user to analysis
USER imbg

# Install miniconda
RUN cd /home/imbg && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/imbg/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add miniconda to path
ENV PATH=/home/imbg/miniconda3/bin:$PATH

# Update base conda env with conda-env yaml file.
# Used for build dependencies for scipy, numpy, etc.
COPY environment.yml /home/imbg/environment.yml
RUN conda env update -n base -f /home/imbg/environment.yml

RUN conda install -n base -c conda-forge hdbscan

# Install fastlbp
RUN pip3 install fastlbp-imbg==0.0.6
