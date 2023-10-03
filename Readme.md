# Model Parallel Minimal Implementation (Fine-tuning LLMs with ZeRO)

This is a minimal implementation for model parallelism in LLMs (large language models).<br>
You can start model parallel implementation by using this template.

In this source code, the pre-trained Meta OPT model (you can choose appropriate size of models) and dataset are downloaded from Hugging Face hub, and this model is then fine-tuned by parallelized manner with PyTorch and DeepSpeed library.

In this source code I have just applied ZeRO for model parallelism, but see [here]( https://tsmatz.wordpress.com/2023/09/21/model-parallelism/) for parallelism overview.

## 1. Set-up and install

### Create a VM

Create a GPU-utilized virtual machine (VM) with "Ubuntu Server 20.04 LTS" image in Microsoft Azure.

In Microsoft Azure, certain VM series - such as, NC and ND-series - have RDMA-capable VMs with SR-IOV and InfiniBand support. Typically, VM SKU with the letter "r" in their name (such as, "Standard_NC24rs_v3") contains the InfiniBand hardware. (However I note that older SKU such like "Standard_NC24r" doesn't support SR-IOV hardware for InfiniBand.)

Select InfiniBand-enabled VM and install drivers, if you need  high-bandwidth inter-node connections, InfiniBand networking. (You can skip this, if you don't need InfiniBand networking.)

### Install GPU driver (CUDA)

Install CUDA (NVIDIA GPU driver) as follows.

```
# compilers and development settings
sudo apt-get update
sudo apt install -y gcc
sudo apt-get install -y make

# install CUDA
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64" >> ~/.bashrc
source ~/.bashrc
```

### Install InfiniBand driver

For installing and setting-up the InfiniBand driver, see "[How to Setup InfiniBand on Azure](https://docs.rapids.ai/deployment/stable/guides/azure/infiniband/)
" (NVIDIA document).

Skip this section if you don't need InfiniBand networking.

### Install NCCL

Install NCCL (NVIDIA Collective Communication Library) as follows.

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

> Note : You can check whether NCCL is enabled in PyTorch, by running ```python3 -c "import torch;print(torch.cuda.nccl.version())"```.

### Install packages

Install PyTorch, Hugging Face (transformer and dataset), and DeepSpeed library as follows.

```
# install and upgrade pip
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
# install packages
pip3 install torch transformers datasets deepspeed
```

> Note : If you prefer, you can also configure MPI to launch jobs in DeepSpeed. (Here I don't use MPI.)


<blockquote>
Note : When you use CPU optimizer (DeepSpeedCPUAdam) in multi-node's training, you need to install with the following options.

```
sudo apt-get install ninja-build
DS_BUILD_CPU_ADAM=1 BUILD_UTILS=1 pip3 install deepspeed
```
</blockquote>

### Setup for multi-node's training

When you run multi-node's training in DeepSpeed, all nodes (participants, remote servers) should be accessible with passwordless SSH.<br>
To setup this environment, please run the following commands in the master node. (Change the following ```{USER_NAME}``` and ```{REMOTE_HOST}``` appropriately.)

```
# create public / private key
ssh-keygen -t rsa -b 4096
# upload public key to remote host
# (repeat for all remote servers)
ssh-copy-id {USER_NAME}@{REMOTE_HOST}
# add identity for logging-in without key auth
eval "$(ssh-agent)"
ssh-add
# deepspeed uses pdsh to launch multi-node training
sudo apt-get install -y pdsh
```

## 2. Configure hostfile

When you run multi-node's training, open ```job/hostfile``` file and configure host name and the number of available GPUs (slots).

## 3. Fine-tune (Train)

Let's run script for fine-tuning.

```single-gpu.py``` is a start point, in which no parallelism is configured.<br>
```zero-optimization.py``` is the parallelized version of the same fine-tuning script.

Before running, change settings in source code (such as, optimizer, model size, batch size, etc), depending on your GPU capacity. (See "```To-Do```" in the source code.)

### Train with a single GPU

```
python3 single-gpu.py
```

### Train with multiple GPUs in a single node

```
deepspeed zero-optimization.py --deepspeed
```

> Note : Available GPUs are automatically detected.

### Train with multiple GPUs in multiple nodes

```
deepspeed --hostfile=job/hostfile zero-optimization.py --deepspeed
```

<blockquote>
Note : To see GPU usage, run nvidia-smi command.

![nvidia-smi execution](https://tsmatz.files.wordpress.com/2023/09/20230925_gpu_usage.jpg)
</blockquote>
