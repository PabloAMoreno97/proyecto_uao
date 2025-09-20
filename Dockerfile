FROM python:3.13

# Instalar dependencias del sistema
RUN apt-get update -y && \
    apt-get install -y python3-opencv python3-tk tk-dev

WORKDIR /home/src

COPY . ./

RUN pip install -r requirements.txt
