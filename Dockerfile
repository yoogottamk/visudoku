FROM 3.6-slim-buster

MAINTAINER yoogottamk "yoogottamk@outlook.com"

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

RUN python digit-recog/data-augment.py && \
    python digit-recog/gen_data.py && \
    python digit-recog/train.py

RUN make && \
    cd sudoku-solver && \
    make

WORKDIR /app/flask-app

ENTRYPOINT [ "python" ]
CMD [ "server.py" ]
