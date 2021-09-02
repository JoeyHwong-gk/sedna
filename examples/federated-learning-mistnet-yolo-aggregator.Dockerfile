FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib
COPY ./plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet  /home/work/

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]
