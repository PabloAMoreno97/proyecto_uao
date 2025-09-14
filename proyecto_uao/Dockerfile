FROM python:3.10

RUN apt-get update -y && \
    apt-get install -y python3-opencv

WORKDIR /home/src

COPY . ./

RUN pip install -r requirements.txt
