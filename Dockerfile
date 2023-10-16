FROM jupyter/base-notebook
USER root
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
COPY requirements.txt /home/jovyan/
RUN pip install --no-cache-dir -r /home/jovyan/requirements.txt
RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ultralytics==8.0.89

