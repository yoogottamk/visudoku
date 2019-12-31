FROM python:3.7-slim

MAINTAINER yoogottamk "yoogottamk@outlook.com"

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get update && \
	apt-get -y install libglib2.0 libsm6 libxext6 libxrender1 make --no-install-recommends

RUN pip3 install -r requirements.txt

COPY . /app

RUN python3 digit-recog/data_augment.py && \
    python3 digit-recog/gen_data.py && \
    python3 digit-recog/train.py

RUN make && \
    cd sudoku-solver && \
    make

WORKDIR /app/flask-app

ENTRYPOINT [ "python3" ]
CMD [ "server.py" ]
