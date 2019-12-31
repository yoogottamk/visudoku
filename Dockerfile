FROM python:3.7-slim

MAINTAINER yoogottamk "yoogottamk@outlook.com"

RUN apt-get update && \
	apt-get -y install libglib2.0 libsm6 libxext6 libxrender1 --no-install-recommends

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

RUN python3 digit-recog/data_augment.py && \
    python3 digit-recog/gen_data.py && \
    python3 digit-recog/train.py

WORKDIR /app/flask-app

ENTRYPOINT [ "python3" ]
CMD [ "server.py" ]
