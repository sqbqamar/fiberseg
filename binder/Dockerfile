FROM python:3.8.16
WORKDIR /app
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ultralytics==8.0.89
COPY . .
CMD ["python", "app.py"]