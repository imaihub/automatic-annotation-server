FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA True
ENV CUDA_HOME /usr/local/cuda-12.4/
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

COPY . /GroundingDINO/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:* \
    libsm6=2:* \
    libxext6=2:* \
    git=1:* \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    vim=2:* \
    libgl1 \
    curl \
    wget \
    build-essential

WORKDIR /GroundingDINO

RUN python -m pip install -r requirements_pinned.txt

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -nc

ENV PYTHONPATH=/GroundingDINO/

CMD ["uvicorn", "grounding_dino_server:app", "--host", "0.0.0.0", "--port", "8901"]
