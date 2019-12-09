from tensorflow/tensorflow:1.15.0-py3 as nsga-keras

WORKDIR /root/nsga-keras
ENV LC_ALL C.UTF-8

COPY install_deps.sh ./
RUN ./install_deps.sh

COPY requirements.txt install_py_deps.sh ./
RUN ./install_py_deps.sh
