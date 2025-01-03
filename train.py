import argparse
import math
import os
import yaml

from tqdm import tqdm
from dotenv import load_dotenv
import wandb

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_scheduler,
)
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)

from src.data import load_data
from src.tokenizer import AudioTokenizer, get_start_tokens
from src.utils import save_checkpoint, get_exp_name

# Parse arguments
parser = argparse.ArgumentParser(description="Train a model with configuration.")
parser.add_argument(
    "--config", type=str, help="Path to the config.yaml file", required=True
)
args = parser.parse_args()

# Load config
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

base_model = config["base_model"]
checkpoint_path = config.get("checkpoint_path")
save_dir = config["save_dir"]

data = config["data"]

start_audio_token = config["start_audio_token"]
end_audio_token = config["end_audio_token"]

path_to_cache = config["path_to_cache"]
checkpointing_steps = int(config["checkpointing_steps"])

max_grad_norm = float(config["max_grad_norm"])
freeze_params = config["freeze"]

torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

load_dotenv()
wandb.login(key=os.getenv("WB_KEY"))


def train(
    model,
    dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
    completed_steps,
    progress_bar,
    max_train_steps,
    save_dir,
):
    model.train()
    total_loss = 0
    acc_loss = 0

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            last_loss = loss.detach().float()
            total_loss += last_loss
            acc_loss += last_loss

            accelerator.backward(loss)

            del batch, loss, outputs
            torch.cuda.empty_cache()

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1

            acc_loss = acc_loss / int(config["gradient_accumulation_steps"])
            accelerator.log({"loss": acc_loss.item()})
            acc_loss = 0

            if completed_steps % checkpointing_steps == 0:
                save_checkpoint(
                    model,
                    accelerator,
                    tokenizer,
                    optimizer,
                    lr_scheduler,
                    save_dir,
                    checkpointing_steps,
                )

            torch.cuda.empty_cache()

        if completed_steps >= max_train_steps:
            break

    return total_loss / len(dataloader), completed_steps


def eval(
    model,
    dataloader,
    accelerator,
    epoch,
    completed_steps,
    train_loss,
):
    model.eval()
    losses = []

    eval_progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", leave=False)

    for batch in eval_progress_bar:
        with torch.no_grad():
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(int(config["eval_batch_size"]))
                )
            )

            del outputs

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

    base_log = {
        "perplexity": perplexity,
        "eval_loss": eval_loss.item(),
        "train_loss": train_loss.item(),
        "epoch": epoch,
        "step": completed_steps,
    }

    accelerator.log(base_log, step=completed_steps)


if __name__ == "__main__":
    import datetime

    timeout = datetime.timedelta(seconds=100000000)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="no",
        log_with="wandb",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False),
            InitProcessGroupKwargs(timeout=timeout),
        ],
    )
    device = accelerator.device

    exp_save_dir = os.path.join(save_dir, get_exp_name(config))
    os.makedirs(exp_save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)
    if checkpoint_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )

    model.gradient_checkpointing_enable()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]"}
        )  # '[PAD]' is the new padding token
        tokenizer.pad_token = "[PAD]"
        config["n_special_tokens"] += 1

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_audio_token, end_audio_token]}
    )
    n_tokens = len(tokenizer)
    print("Not audio tokens:", n_tokens)

    start_audio_token_id = tokenizer(start_audio_token)["input_ids"][-1]
    end_audio_token_id = tokenizer(end_audio_token)["input_ids"][-1]

    tokens_config = get_start_tokens(config["quantizer"], n_tokens)
    quantizer = AudioTokenizer(config["quantizer"], tokens_config)

    codebook_size = (
        config["quantizer"]["speech"]["n_new_tokens"]
        + config["quantizer"]["wav"]["n_new_tokens"]
    )
    print("New tokens:", codebook_size)
    train_dataset, val_dataset = load_data(data, tokenizer, quantizer, config)

    model.resize_token_embeddings(n_tokens + codebook_size)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=int(config["train_batch_size"]),
        num_workers=16,
    )
    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=int(config["eval_batch_size"]),
        num_workers=16,
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": float(config["weight_decay"]),
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=float(config["learning_rate"]),  # fused=True
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / int(config["gradient_accumulation_steps"])
    )
    max_train_steps = int(config["num_train_epochs"]) * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(config["num_warmup_steps"]) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    if checkpoint_path is not None and os.path.exists(
        os.path.join(checkpoint_path, "optimizer.pt")
    ):
        optim_state = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
        scheduler_state = torch.load(os.path.join(checkpoint_path, "scheduler.pt"))

        optimizer.load_state_dict(optim_state)
        lr_scheduler.load_state_dict(scheduler_state)

    # model = freeze(model, freeze_other=False, freeze_ff=True, freeze_ff_layers=[31])

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / int(config["gradient_accumulation_steps"])
    )
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    accelerator.init_trackers(
        config["wandb_project_name"],
        {"lr_scheduler_type": config["lr_scheduler_type"]},
    )

    total_batch_size = (
        config["train_batch_size"]
        * accelerator.num_processes
        * int(config["gradient_accumulation_steps"])
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {config['train_batch_size']}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
    print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, num_train_epochs):
        train_loss, completed_steps = train(
            model,
            train_dataloader,
            accelerator,
            optimizer,
            lr_scheduler,
            completed_steps,
            progress_bar,
            max_train_steps,
            exp_save_dir,
        )
        print(f"EPOCH {epoch + 1} train loss:", train_loss)
        eval(
            model,
            eval_dataloader,
            accelerator,
            epoch,
            completed_steps,
            train_loss,
        )

    save_checkpoint(
        model,
        accelerator,
        tokenizer,
        optimizer,
        lr_scheduler,
        exp_save_dir,
        checkpointing_steps,
    )
