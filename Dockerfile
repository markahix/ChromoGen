# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
FROM continuumio/miniconda3

## System Installations
RUN apt-get update
RUN apt-get install -y --no-install-recommends libatlas-base-dev gfortran git

## Set python version
RUN conda install -y python==3.8

## Install necessary libraries
RUN pip install torch rdkit chemprop transformers xgboost seaborn molsets==0.1.0 git+https://github.com/reymond-group/map4@v1.0
RUN conda install -c tmap tmap
RUN conda install -y -c conda-forge vina 
RUN pip install networkx==2.8.8 meeko

## Install application
COPY ./ /app
WORKDIR /app/data/gdb/gdb13/

RUN wget https://zenodo.org/record/5172018/files/gdb13.1M.freq.ll.smi.gz
RUN gzip -d gdb13.1M.freq.ll.smi.gz && mv gdb13.1M.freq.ll.smi gdb13.smi

WORKDIR /app