FROM jupyter/base-notebook
USER root
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y

# Install Python 3.8 using conda
RUN conda install --quiet --yes python=3.8

# Create a new Conda environment and activate it
RUN conda create --quiet --yes --name myenv python=3.8
RUN echo "conda activate myenv" >> ~/.bashrc

# Set the default working directory
WORKDIR /home/jovyan

# Install additional packages (for example, from your requirements.txt file)
COPY requirements.txt /home/jovyan/
RUN pip install --no-cache-dir -r requirements.txt

