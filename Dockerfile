FROM jupyter/base-notebook
USER root
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
COPY requirements.txt /home/jovyan/
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt


