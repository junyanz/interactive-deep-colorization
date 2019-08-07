FROM continuumio/miniconda3

COPY ./install/docker_conda_install.sh /app/install/
WORKDIR /app/install/

RUN pwd
RUN ls

RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN ls
RUN ./docker_conda_install.sh

COPY . /app
WORKDIR /app

CMD ["python","ideepcolor_docker.py"]
