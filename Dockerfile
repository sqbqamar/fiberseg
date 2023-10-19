FROM jupyter/base-notebook
USER root
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y

# Install Python 3.8 using conda
#RUN conda install --quiet --yes python=3.8

# Create a new Conda environment and activate it
#RUN conda create --quiet --yes --name myenv python=3.8
#RUN echo "conda activate myenv" >> ~/.bashrc

# Set the default working directory
WORKDIR /home/jovyan

# Install additional packages ( from requirements.txt file)
COPY requirements.txt /home/jovyan/
COPY prediction_file.ipynb /home/jovyan/
RUN pip install --no-cache-dir -r requirements.txt
#RUN conda install pytorch torchvision
#RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.15.1
#RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
#RUN pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ultralytics==8.0.89

