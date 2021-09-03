FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx git

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib
RUN git clone https://github.com/TL-System/plato.git /home/plato

RUN pip install -r /home/plato/requirements.txt
RUN pip install -r /home/plato/packages/yolov5/requirements.txt

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]
