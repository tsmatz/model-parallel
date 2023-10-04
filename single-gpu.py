"""
To-Do :
Please change the following settings depending on your capacity (the size of GPUs).
"""

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
from datasets import load_dataset
import math

# To-Do : Select model size
# model_name = "facebook/opt-6.7b"
model_name = "facebook/opt-1.3b"
# model_name = "facebook/opt-350m"
# model_name = "facebook/opt-125m"

dataset_name = "Dahoas/rm-static"
num_epochs = 2
# To-Do : Set batch size and the size for gradient accumulation steps
batch_size = 16
gradient_accumulation_steps = 8

# To-Do : Select device (GPU or CPU)
device = torch.device("cuda")
#device = torch.device("cpu")

# Load model
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
)
model = model.to(device)

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
    all_tokens = tokenizer(
        [data["prompt"][i]+data["chosen"][i] for i in range(len(data["prompt"]))],
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt")
    return {
        "input_ids": all_tokens["input_ids"],
        "attention_mask": all_tokens["attention_mask"],
        "labels": all_tokens["input_ids"],
    }
dataset = dataset.map(
    _align_dataset,
    remove_columns=["prompt", "response", "chosen", "rejected"],
    batched=True,
    batch_size=128)

# Here we use only train dataset
dataset = dataset["train"]

# Create PyTorch dataloader
sampler = RandomSampler(dataset)
dataloader = DataLoader(
    dataset,
    collate_fn=default_data_collator,
    sampler=sampler,
    batch_size=batch_size)

# Set up optimizer
optimizer = torch.optim.AdamW(
    params=model.parameters(),  # divide into weight_decay and no_weight_decay if needed
    lr=1e-3,
    betas=(0.9, 0.95),
)

# Build cosine scheduler
num_update_steps = math.ceil(len(dataloader) / batch_size / gradient_accumulation_steps)
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

# Run train loop
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(dataloader):
        batch = {k:v.to(device) for k, v in batch.items()}  # send to GPU
        with torch.set_grad_enabled(True):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            print(f"Epoch {epoch+1} {math.ceil((i + 1) / gradient_accumulation_steps)}/{num_update_steps} - loss: {loss :2.4f}", end="\r")
    print("")
