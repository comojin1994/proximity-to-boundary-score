FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.1-cuda12.1.0

USER root

RUN apt-get -y update --fix-missing
RUN apt-get install -y git
RUN apt-get install -y curl

WORKDIR /workspace
RUN chmod -R a+w /workspace

# Application
COPY requirements/torch-train.txt /workspace/requirements.txt

RUN pip install --ignore-installed -r requirements.txt