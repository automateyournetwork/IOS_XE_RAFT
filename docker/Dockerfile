FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo \
 && sudo apt-get update -y

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y \
 && apt install openssh-client -y

RUN echo "==> Adding pyATS ..." \
 && pip install pyats[full]

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
  && pip install -U --quiet langchain_experimental langchain langchain-community \
  && pip install chromadb \
  && pip install tiktoken

RUN echo "==> Install jq.." \
  && pip install jq

RUN echo "==> Adding InstructorEmbedding ..." \
  && pip install -U sentence-transformers==2.2.2 \
  && pip install InstructorEmbedding

RUN echo "==> Install wandb.." \
  && pip install wandb

RUN echo "==> Install dataset.." \
  && pip install dataset

RUN echo "==> Install peft.." \
  && pip install peft

RUN echo "==> Install accelerate.." \
  && pip install accelerate \
  && pip install -i https://pypi.org/simple/ bitsandbytes


COPY /routing_table_raft /routing_table_raft/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]