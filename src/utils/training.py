import os

import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


def freeze(
    model,
    freeze_emb=False,
    freeze_ln=True,
    freeze_attn=True,
    freeze_ff=True,
    freeze_ff_layers=None,  # None means all or no layers, depending on freeze_ff
    freeze_other=True,
):
    if freeze_ff_layers is not None and not isinstance(freeze_ff_layers, (list, set)):
        raise ValueError("freeze_ff_layers must be a list or set of layer indices")

    for name, p in model.named_parameters():
        name = name.lower()
        layer_index = None
        if "mlp" in name:
            # Parse the layer index from the parameter name if possible
            tokens = name.split(".")
            for token in tokens:
                if token.isdigit():
                    layer_index = int(token)
                    break

        if "ln" in name or "norm" in name:
            p.requires_grad = not freeze_ln
        elif "embeddings" in name:
            p.requires_grad = not freeze_emb
        elif "mlp" in name:
            if freeze_ff_layers is None:
                # Apply general freeze_ff setting
                p.requires_grad = not freeze_ff
            else:
                # Apply specific layer freeze setting
                p.requires_grad = not (freeze_ff and layer_index in freeze_ff_layers)
        elif "attn" in name:
            # if not p.requires_grad or freeze_attn:
            print("attn", name)
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model


def get_last_checkpoint(save_dir):
    n_checkpoints = len(
        list(filter(lambda x: x.startswith("checkpoint"), os.listdir(save_dir)))
    )
    return n_checkpoints + 1


def save_checkpoint(
    model, accelerator, tokenizer, optimizer, scheduler, save_dir, checkpointing_steps
):
    accelerator.wait_for_everyone()

    path = os.path.join(
        save_dir, f"checkpoint-{get_last_checkpoint(save_dir) * checkpointing_steps}"
    )

    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)

    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT
        ):
            state_dict = model.state_dict()
            if accelerator.is_main_process:
                torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            path,
            is_main_process=accelerator.is_main_process,
            safe_serialization=False,
        )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(path)
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))


def left_pad_sequence(sequences, padding_value=0):
    # Find the maximum length in the batch
    max_length = max(seq.size(0) for seq in sequences)

    # Apply left-padding to each sequence
    padded_sequences = [
        torch.cat(
            [
                torch.full((max_length - seq.size(0), *seq.shape[1:]), padding_value),
                seq,
            ],
            dim=0,
        )
        for seq in sequences
    ]
    return torch.stack(padded_sequences)


def collate_fn(batch, tokenizer, max_seq_length=512):
    # Truncate sequences to max_seq_length
    truncated_batch = []
    for item in batch:
        input_ids = item["input_ids"][:max_seq_length]
        attention_mask = item["attention_mask"][:max_seq_length]
        labels = item["labels"][:max_seq_length]
        truncated_batch.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    # Extract input_ids, attention_masks, and labels from the truncated batch
    input_ids = [item["input_ids"] for item in truncated_batch]
    attention_masks = [item["attention_mask"] for item in truncated_batch]
    labels = [item["labels"] for item in truncated_batch]

    # Pad sequences to the max length in the batch
    input_ids = left_pad_sequence(input_ids, padding_value=tokenizer.pad_token_id)
    attention_masks = left_pad_sequence(attention_masks, padding_value=0)
    labels = left_pad_sequence(labels, padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


def get_exp_name(config):
    name = config["base_model"].split("/")[-1]

    if "asr" in config["tasks"]:
        name += "_asr"
        for aq in config["quantizer"]["asr"]:
            name += f"_{aq['quantizer']}_{aq['n_codebooks']}"

    if "tts" in config["tasks"]:
        name += "_tts"
        for tq in config["quantizer"]["tts"]:
            name += f"_{tq['quantizer']}_{tq['n_codebooks']}"

    if len(config["text_data"]):
        name += f"_text"

    return name
