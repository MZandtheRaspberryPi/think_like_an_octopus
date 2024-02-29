FROM python:3.9.18-slim-bullseye

RUN apt-get update && apt-get install -y procps nano

# cpp stuff
RUN apt-get install g++ cmake -y

COPY requirements.txt /problem/requirements.txt

# python stuff
RUN pip install -r /problem/requirements.txt

COPY ./pf_sector_helpers.cpp /problem/pf_sector_helpers.cpp
RUN mkdir /problem/build
RUN g++ -fPIC -pthread -shared -o /problem/build/pf_sector_helpers.so /problem/pf_sector_helpers.cpp -std=c++11
RUN g++ -fPIC -pthread -o /problem/build/pf_sector_helpers.exe /problem/pf_sector_helpers.cpp -std=c++11
RUN cp /problem/build/pf_sector_helpers.so /problem/pf_sector_helpers.so
RUN cp /problem/build/pf_sector_helpers.exe /problem/pf_sector_helpers.exe

ENV PYTHONPATH=/problem

COPY main.py /problem/main.py

WORKDIR /problem
ENTRYPOINT ["/bin/bash"]