from src.utils import (
    freeze,
    freeze_entire_model,
    get_audio_padding_tokens,
    decode_audio,
    prepare_librispeech,
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

n_codebooks = config["n_codebooks"]
max_seq_length = config["max_seq_length"]

load_processed = config["load_processed"]
path_to_processed = config["path_to_processed"]
path_to_cache = config["path_to_cache"]
quantize_before_training = config["quantize_before_training"]

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
            aduio.write(audio_file)
            audios.append(audio_file)
        except:
            print("No audio generated.")
            pass

    return audios


def get_last_checkpoint():
    n_checkpoints = len(
        list(filter(lambda x: x.startswith("checkpoint"), os.listdir(save_dir)))
    )
    return n_checkpoints + 1


def save_checkpoint(model, accelerator, tokenizer, optimizer, scheduler):
    accelerator.wait_for_everyone()
    state = model.state_dict()

    path = os.path.join(
        save_dir, f"checkpoint-{get_last_checkpoint() * checkpointing_steps}"
    )

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        path,
        state_dict=state,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        save_embedding_layers=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(path)
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))


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
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="no",
        log_with="wandb",
    )

    os.makedirs(save_dir, exist_ok=True)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, collate_fn=default_data_collator, batch_size=eval_batch_size
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
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
        optimizer_grouped_parameters, lr=learning_rate, fused=True
    )
    # optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, min_8bit_size=16384)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * accelerator.num_processes,
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
        len(train_dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    accelerator.init_trackers(
        "vikhr4o-llama-tiny", {"lr_scheduler_type": lr_scheduler_type}
    )

    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
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
