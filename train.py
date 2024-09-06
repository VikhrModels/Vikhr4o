from src.utils import (
    freeze,
    get_audio_padding_tokens,
    decode_audio,
    save_checkpoint
)
from src.data import load_data

import math
import random

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_scheduler,
)
from accelerate import Accelerator, DistributedDataParallelKwargs,InitProcessGroupKwargs

import argparse
import yaml
from speechtokenizer import SpeechTokenizer
import os


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
save_dir = config["save_dir"]

data = config["data"]

start_audio_token = config["start_audio_token"]
end_audio_token = config["end_audio_token"]
end_sequence_token = config["end_sequence_token"]
n_special_tokens = config["n_special_tokens"]

n_codebooks_tts = int(config["n_codebooks_tts"])
n_codebooks_asr = int(config["n_codebooks_asr"])
max_seq_length = int(config["max_seq_length"])

load_processed = bool(config["load_processed"])
path_to_processed = config["path_to_processed"]
path_to_cache = config["path_to_cache"]
quantize_before_training = bool(config["quantize_before_training"])
checkpointing_steps = int(config['checkpointing_steps'])
max_grad_norm = float(config['max_grad_norm'])
torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
torch.backends.cudnn.allow_tf32 = config["allow_tf32"]


def test_audio_generation(model, batch, n, quantizer, pad_tokens, n_original_tokens):
    inds = random.choices(range(len(batch)), k=n)
    audios = []

    for input_ids, attn in batch["input_ids"], batch["attention_mask"]:
        with torch.no_grad():
            ind = torch.nonzero(input_ids == start_audio_token_id)[0, -1]
            input_ids = input_ids[: ind + 1].unsqueeze(0)
            attn = attn[: ind + 1].unsqueeze(0).to(torch.float16)
            output = model.generate(
                input_ids=input_ids, attention_mask=attn, max_length=max_seq_length
            )

        try:
            audio = decode_audio(output, quantizer, pad_tokens, n_original_tokens)
            audio_file = os.path.join(save_dir, "audio")
            os.makedirs(audio_file, exists_ok=True)
            audio_file = os.path.join(audio_file, f"audio_{ind + 1}.wav")
            audio.write(audio_file)
            audios.append(audio_file)
        except:
            print("No audio generated.")
            pass

    return audios


def train(
    model,
    dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
    completed_steps,
    progress_bar,
    max_train_steps,
    quantizer,
    n_codebooks_tts,
    n_codebooks_asr,
    max_seq_length,
    n_special_tokens,
    device,
):
    model.train()
    total_loss = 0
    acc_loss = 0
    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
    eos = tokenizer(end_sequence_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            # TODO: won't work for batch_size > 1, need to change
            # Quantization
            audio_data, sample_rate = batch["audio_data"], batch["sampling_rate"]

            if batch["asr"][0]:
                n_codebooks = n_codebooks_asr
            else:
                n_codebooks = n_codebooks_tts
            
            audio = torch.tensor(audio_data).view(1, 1, len(audio_data[0])).float()
            audio = audio.to(device)
            codes = quantizer.encode(audio)
            codes = codes.squeeze(1)

            text_input_tokens = batch["text_input_tokens"].to(device)
            raw_audio_tokens = codes[:n_codebooks]
            
            audio_input_tokens = raw_audio_tokens.t().contiguous().view(1, -1)
            audio_length = min(
                max_seq_length - text_input_tokens.shape[-1] - n_special_tokens,
                audio_input_tokens.shape[-1],
            )
            audio_length -= audio_length % n_codebooks
            padding_size = (
                max_seq_length
                - text_input_tokens.shape[-1]
                - audio_length
                - n_special_tokens
            )
            padding = torch.zeros((1, padding_size), dtype=torch.int64, device=device)
           
            if batch["asr"][0]:
                tokens = torch.cat(
                    [
                        padding,
                        soa,
                        audio_input_tokens[:, :audio_length],
                        eoa,
                        text_input_tokens.squeeze(1),
                        eos,
                    ],
                    dim=1,
                )
            else:
                tokens = torch.cat(
                    [
                        padding,
                        text_input_tokens.squeeze(1),
                        soa,
                        audio_input_tokens[:, :audio_length],
                        eoa,
                        eos,
                    ],
                    dim=1,
                )

            attention_mask = torch.cat(
                [padding, torch.ones((1, max_seq_length - padding_size), device=device)],
                dim=1,
            )

            if tokens.shape[1] > max_seq_length:
                continue

            batch = {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "labels": tokens.clone(),
            }

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            last_loss = loss.float()
            total_loss += last_loss
            acc_loss += last_loss

            accelerator.backward(loss)

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
                save_checkpoint(model, accelerator, tokenizer, optimizer, lr_scheduler, save_dir, completed_steps)

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
    quantizer,
    n_codebooks,
    max_seq_length,
    n_special_tokens,
    device,
):
    model.eval()
    losses = []
    soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
    eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
    eos = tokenizer(end_sequence_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
    
    eval_progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", leave=False)

    for batch in eval_progress_bar:
        with torch.no_grad():
            # Quantization
            audio_data, sample_rate = batch["audio_data"], batch["sampling_rate"]
            audio = torch.tensor(audio_data).view(1, 1, len(audio_data)).float()
            audio = audio.to(device)
            codes = quantizer.encode(audio)
            codes = codes.squeeze(1)
            del audio
            torch.cuda.empty_cache()

            text_input_tokens = batch["text_input_tokens"].to(device)
            raw_audio_tokens = codes[:n_codebooks]

            audio_input_tokens = raw_audio_tokens.t().contiguous().view(1, -1)
            audio_length = min(
                max_seq_length - text_input_tokens.shape[-1] - n_special_tokens,
                audio_input_tokens.shape[-1],
            )
            audio_length -= audio_length % n_codebooks
            padding_size = (
                max_seq_length
                - text_input_tokens.shape[-1]
                - audio_length
                - n_special_tokens
            )
            padding = torch.zeros((1, padding_size), dtype=torch.int64, device=device)

            tokens = torch.cat(
                [
                    padding,
                    text_input_tokens,
                    soa,
                    audio_input_tokens[:, :audio_length],
                    eoa,
                    eos,
                ],
                dim=1,
            ).squeeze(0)

            attention_mask = torch.cat(
                [padding, torch.ones((1, max_seq_length - padding_size), device=device)],
                dim=1,
            ).squeeze(0)

            if tokens.shape[1] > max_seq_length:
                continue

            batch = {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "labels": tokens.clone(),
            }

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(int(config["eval_batch_size"]))))
    
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
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True), InitProcessGroupKwargs(timeout=timeout)],
    )
    device = accelerator.device
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, attn_implementation="sdpa", torch_dtype=torch.bfloat16, cache_dir=path_to_cache
    )
    model.gradient_checkpointing_enable()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_audio_token, end_audio_token]}
    )
    n_tokens = len(tokenizer)

    start_audio_token_id = tokenizer(start_audio_token)["input_ids"][-1]
    end_audio_token_id = tokenizer(end_audio_token)["input_ids"][-1]

    config_path = config["quantizer_config_path"]
    ckpt_path = config["quantizer_ckpt_path"]
    quantizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    quantizer.eval().to(device)

    codebook_size = quantizer.quantizer.bins

    tokenizer.add_tokens([f"<audio_token_{i}>" for i in range(codebook_size)])

    assert len(tokenizer) == n_tokens + codebook_size

    model.resize_token_embeddings(len(tokenizer))

    if not load_processed:
        train_dataset, val_dataset = load_data(data, tokenizer, path_to_cache)
    else:
        train_dataset = load_from_disk(os.path.join(path_to_processed, "train"))
        val_dataset = load_from_disk(os.path.join(path_to_processed, "val"))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=int(config["train_batch_size"]),
        num_workers=16
    )
    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=int(config["eval_batch_size"]), 
        num_workers=16
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
        optimizer_grouped_parameters, lr=float(config["learning_rate"]),  # fused=True
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
        config["wandb_project_name"], {"lr_scheduler_type": config["lr_scheduler_type"]}
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
    padding_tokens = get_audio_padding_tokens(quantizer, device)
    
    model = freeze(model)
  
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
            quantizer,
            n_codebooks_tts,
            n_codebooks_asr,
            max_seq_length,
            n_special_tokens, 
            device
        )
        print(f"EPOCH {epoch + 1} train loss:", train_loss)
        eval(
            model,
            eval_dataloader,
            accelerator,
            epoch,
            completed_steps,
            train_loss,
            quantizer,
            n_tokens + 1,
            max_seq_length,
            n_special_tokens,
            device
        )

    save_checkpoint(model, accelerator, tokenizer, optimizer, lr_scheduler,  save_dir, epoch)
