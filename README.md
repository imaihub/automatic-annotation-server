# Automatic Annotation Server

This repository will consist of multiple generic models that can provide initial annotations for a dataset. Currently, only GroundingDINO is supported.

## GroundingDINO

### Install with Docker

First install docker using: 

```bash
sudo apt install docker.io
sudo apt install docker-compose
```

If you wish to run the models on the GPU, install nvidia-container-toolkit following the instructions from (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or execute

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y nvidia-docker2

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-compose-plugin

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

To run this server locally in a Docker container, run

```bash
sudo docker compose up
```

### Install without Docker

#### Ubuntu

CUDA needs to be installed and CUDA_HOME needs to be set to that directory (often somewhere like usr/local/cuda-12.6)

To install CUDA Toolkit 12.6 on Ubuntu 24.04, run the following:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

You should set the environment variable manually as follows if you want to build a local GPU environment:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-12.6/ # Update if using another CUDA version
```

Install the package requirements
```bash
pip install -r requirements_pinned.txt
```

Install Grounding DINO:
```bash
pip install --no-build-isolation -e GroundingDINO
```

Download the weights
```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Start the fastapi server:
```bash
uvicorn grounding_dino_server:app --host 0.0.0.0 --port 8901
```
