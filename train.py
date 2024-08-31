from src.utils import (
    freeze,
    freeze_entire_model,
    get_audio_padding_tokens,
    decode_audio,
    prepare_librispeech,
    prepare_tedlium,
    prepare_parler_tts,
    prepare_synthetic,
)

import math
import random
import wandb

from tqdm import tqdm
from multiprocess import set_start_method

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk, Audio, concatenate_datasets
from datasets import Value, DatasetDict


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)
from accelerate import Accelerator
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

n_codebooks_tts = config["n_codebooks_tts"]
n_codebooks_asr = config["n_codebooks_asr"]
max_seq_length = config["max_seq_length"]

load_processed = config["load_processed"]
path_to_processed = config["path_to_processed"]
path_to_cache = config["path_to_cache"]
quantize_before_training = config["quantize_before_training"]

torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
torch.backends.cudnn.allow_tf32 = config["allow_tf32"]


class Vikhr4oDataset(Dataset):
    def __init__(self, dataset, tokenizer, quantizer, device, asr=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.asr = asr
        
        if asr:
            self.n_codebooks = n_codebooks_asr
        else:
            self.n_codebooks = n_codebooks_tts

        self.soa = tokenizer(start_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
        self.eoa = tokenizer(end_audio_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)
        self.eos = tokenizer(end_sequence_token, return_tensors="pt")["input_ids"][
            :, -1:
        ].to(device)

        self.n_original_tokens = len(tokenizer) - 1024
        self.quantizer = quantizer

    def __len__(self):
        return len(self.dataset)

    def quantize(self, example):
        audio_data, sample_rate = (
            example["audio"]["array"],
            example["audio"]["sampling_rate"],
        )
        audio = torch.tensor(audio_data).view(1, 1, len(audio_data)).float()
        audio = audio.to(device)
        codes = self.quantizer.encode(audio)
        codes = codes.squeeze(1)
        del audio
        torch.cuda.empty_cache()
        return codes + self.n_original_tokens

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row["text"]
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"].to(device)
        codes = self.quantize(row)

        raw_audio_tokens = codes[:self.n_codebooks]
        audio_input_tokens = raw_audio_tokens.t().contiguous().view(1, -1)

        audio_length = min(
            max_seq_length - text_input_tokens.shape[-1] - n_special_tokens,
            audio_input_tokens.shape[-1],
        )

        audio_length -= audio_length % self.n_codebooks

        padding_size = (
            max_seq_length
            - text_input_tokens.shape[-1]
            - audio_length
            - n_special_tokens
        )
        padding = torch.zeros((1, padding_size), dtype=torch.int64, device=device)

        if self.asr:
            tokens = torch.cat(
                [
                    padding,
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    text_input_tokens,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)
        else:
            tokens = torch.cat(
                [
                    padding,
                    text_input_tokens,
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)

        attention_mask = torch.cat(
            [padding, torch.ones((1, max_seq_length - padding_size), device=device)],
            dim=1,
        ).squeeze(0)

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": tokens.clone(),
        }


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
            aduio.write(audio_file)
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
):
    model.gradient_checkpointing_enable()
    model.train()
    # model = freeze(model, freeze_ff_layers=None)
    total_loss = 0
    acc_loss = 0

    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            last_loss = loss.detach().float()
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

            acc_loss = acc_loss / gradient_accumulation_steps

            accelerator.log({"loss": acc_loss.item()})
            acc_loss = 0

            if completed_steps % checkpointing_steps == 0:
                save_checkpoint(model, accelerator, tokenizer, optimizer, lr_scheduler)

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
    pad_tokens,
    n_original_tokens,
):
    model.eval()
    losses = []

    eval_progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", leave=False)

    for batch in eval_progress_bar:
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
    # audios = test_audio_generation(model, batch, 2, quantizer, pad_tokens, n_original_tokens)

    base_log = {
        "perplexity": perplexity,
        "eval_loss": eval_loss,
        "train_loss": train_loss.item() / len(train_dataloader),
        "epoch": epoch,
        "step": completed_steps,
    }
    # base_log.update({f"audio_{i+1}": audios[i] for i in range(len(audios))})

    accelerator.log(base_log, step=completed_steps)


if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision="no",
        log_with="wandb",
    )
    device = accelerator.device
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, attn_implementation="sdpa", cache_dir=path_to_cache
    )

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_audio_token, end_audio_token]}
    )
    n_tokens = len(tokenizer)

    start_audio_token_id = tokenizer(start_audio_token)["input_ids"][-1]
    end_audio_token_id = tokenizer(end_audio_token)["input_ids"][-1]

    config_path = config["quantizer_config_path"]
    ckpt_path = config["quantizer_ckpt_path"]
    quantizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    quantizer.eval()

    for n, child in quantizer.named_children():
        child.to(device)
        child = freeze_entire_model(child)

    codebook_size = quantizer.quantizer.bins

    tokenizer.add_tokens([f"<audio_token_{i}>" for i in range(codebook_size)])

    assert len(tokenizer) == n_tokens + codebook_size

    model.resize_token_embeddings(len(tokenizer))

    if not load_processed:
        print("Loadiing data")
        if data == "tedlium":
            dataset = prepare_tedlium()

            train_data = dataset["train"]
            val_data = dataset["validation"]

        elif data == "parler-tts":
            dataset = prepare_parler_tts()

            train_data = dataset["train"]
            val_data = dataset["dev"]

        elif data == "librispeech":
            dataset = prepare_librispeech()

            train_data = dataset["train.100"]
            val_data = dataset["validation"]

        elif data == "synthetic":
            dataset = prepare_synthetic()["train"]

            splits = dataset.train_test_split(test_size=0.1)
            train_data = splits["train"]
            val_data = splits["test"]
    else:
        train_data = load_from_disk(os.path.join(path_to_processed, "train"))
        val_data = load_from_disk(os.path.join(path_to_processed, "val"))

    train_dataset_tts = Vikhr4oDataset(train_data, tokenizer, quantizer, device = device)
    train_dataset_asr = Vikhr4oDataset(train_data, tokenizer, quantizer, device = device, asr=True)

    val_dataset_tts = Vikhr4oDataset(val_data, tokenizer, quantizer, device = device)
    val_dataset_asr = Vikhr4oDataset(val_data, tokenizer, quantizer, device = device, asr=True)

    train_dataset = ConcatDataset([train_dataset_tts, train_dataset_asr])
    val_dataset = ConcatDataset([val_dataset_tts, val_dataset_asr])

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config["train_batch_size"],
    )
    eval_dataloader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=config["eval_batch_size"],
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config["weight_decay"],
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
        optimizer_grouped_parameters, lr=config["learning_rate"], fused=True
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["gradient_accumulation_steps"]
    )
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"] * accelerator.num_processes,
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
        len(train_dataloader) / config["gradient_accumulation_steps"]
    )
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    accelerator.init_trackers(
        config["wandb_project_name"], {"lr_scheduler_type": config["lr_scheduler_type"]}
    )

    total_batch_size = (
        config["train_batch_size"]
        * accelerator.num_processes
        * config["gradient_accumulation_steps"]
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
            padding_tokens,
            n_tokens + 1,
        )

    save_checkpoint(model, accelerator, tokenizer, optimizer, lr_scheduler)
