FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN conda env create --file environment.yml
RUN bash -c "conda init bash"