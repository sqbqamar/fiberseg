FROM binder/base-notebook
RUN apt-get update 
RUN apt install -y libgl1-mesa-glx 