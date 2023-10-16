RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
