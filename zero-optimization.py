"""
To-Do :
Please change the following settings depending on your capacity (the size of GPUs).
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
from datasets import load_dataset
import argparse
import math
import os

import deepspeed
# To-Do : Select optimizer
# (You can offload by using CPU optimizer.)
from deepspeed.ops.adam import FusedAdam
# from deepspeed.ops.adam import DeepSpeedCPUAdam

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Deepspeed launcher passes "local_rank"
parser = argparse.ArgumentParser(description="My training script")
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local rank passed from distributed launcher")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# To-Do : Select model size
model_name = "facebook/opt-6.7b"
#model_name = "facebook/opt-1.3b"
#model_name = "facebook/opt-350m"
#model_name = "facebook/opt-125m"

dataset_name = "Dahoas/rm-static"
num_epochs = 2
# To-Do : Set batch size and the size for gradient accumulation steps
train_micro_batch_size_per_gpu = 16
gradient_accumulation_steps = 8

ds_config = {
    "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "steps_per_print": 10,
    # To-Do : Set precision if needed
    # "fp16": {
    #     "enabled": True,
    # },
    # To-Do : Offload to CPU if needed
    "zero_optimization": {
        "stage": 3,
        # "offload_optimizer": {
        #     "device": "cpu"
        # },
        # "offload_param": {
        #     "device": "cpu"
        # }
        "cpu_offload": False,
  }
}

# Get device
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

# Initialize backend in DeepSpeed
torch.distributed.init_process_group(backend="nccl")
deepspeed.init_distributed()

# Load model
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
)

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    fast_tokenizer=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Here I simply use prompt/response dataset in English
# (Use other dataset for other languages)
dataset = load_dataset(dataset_name)

# Filter dataset not to exceed the length 512
# (model_config.max_position_embeddings is 2048 in OPT)
max_seq_len = 512
dataset = dataset.filter(lambda d: len(d["prompt"] + d["chosen"]) < model_config.max_position_embeddings)

# Convert dataset
def _align_dataset(data):
    prompt_tokens = tokenizer(
        data["prompt"],
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt")
    label_tokens = tokenizer(
        [data["prompt"][i]+data["chosen"][i] for i in range(len(data["prompt"]))],
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt")
    return {
        "input_ids": prompt_tokens["input_ids"],
        "attention_mask": prompt_tokens["attention_mask"],
        "labels": label_tokens["input_ids"],
    }
dataset = dataset.map(
    _align_dataset,
    remove_columns=["prompt", "response", "chosen", "rejected"],
    batched=True,
    batch_size=128)

# Here we use only train dataset
dataset = dataset["train"]

# Set up optimizer manually
# (You can also set up optimizer with DeepSpeed configuration.)

# To-Do : Select optimizer
# (You can offload by using CPU optimizer.)
optimizer = FusedAdam(
    params=model.parameters(),  # divide into weight_decay and no_weight_decay if needed
    lr=1e-3,
    betas=(0.9, 0.95))
# optimizer = DeepSpeedCPUAdam(
#     model_params=model.parameters(),  # divide into weight_decay and no_weight_decay if needed
#     lr=1e-3,
#     betas=(0.9, 0.95))

# Build cosine scheduler manually
# (You can also use DeepSpeed built-in scheduler in configuration.)
#####num_update_steps = math.ceil(len(dataloader) / train_micro_batch_size_per_gpu / gradient_accumulation_steps)
num_update_steps = math.ceil(len(dataset["input_ids"]) / train_micro_batch_size_per_gpu / gradient_accumulation_steps)
def _get_cosine_schedule(
    current_step: int,
    num_warmup_steps: int = 0,
    num_training_steps: int = num_epochs * num_update_steps
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
scheduler = LambdaLR(optimizer, lr_lambda=_get_cosine_schedule)

# Initialize components (except for scheduler)
model_engine, optimizer, dataloader, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    optimizer=optimizer,
    args=args,
    # lr_scheduler=scheduler,  # see below note
    training_data=dataset,
    collate_fn=default_data_collator,
    dist_init_required=True
)

# Run train loop
for epoch in range(num_epochs):
    model_engine.train()
    for i, batch in enumerate(dataloader):
        batch = {k:v.to(device) for k, v in batch.items()}  # send to GPU
        with torch.set_grad_enabled(True):
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            # Note : Here I manually step scheduler,
            # but you can also set scheduler in deepspeed.initialize()
            # when it's supposed to execute at every training step.
            if model_engine.is_gradient_accumulation_boundary():
                scheduler.step()
            print(f"Epoch {epoch+1} {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_update_steps} - loss: {loss :2.4f}", end="\r")
    print("")
