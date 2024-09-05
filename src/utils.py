import torch
from audiotools import AudioSignal
import os
from datasets import load_dataset, Dataset
from datasets import Value


def freeze_entire_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    return model


'''
5,
        6,
        7,
        8,
        9,
        12,
        23,
        14,
        18,
        19,
        20,
        0,
        25,
'''


def freeze(
        model,
        freeze_emb=False,
        freeze_ln=True,
        freeze_attn=True,
        freeze_ff=True,
        freeze_ff_layers=[],  # None means all or no layers, depending on freeze_ff
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
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model


def get_audio_padding_tokens(quantizer, device):
    # create audio without any sounds
    # seems to work better than random padding if
    # length of generated audio is not divisible by n_codebooks
    audio = torch.zeros((1, 1, 1))
    audio = audio.to(device)

    codes = quantizer.encode(audio)

    # Move tensor back to CPU and delete it to free GPU memory
    del audio
    torch.cuda.empty_cache()

    return {"audio_tokens": codes.squeeze(1)}


def decode_audio(
        tokens,
        quantizer,
        pad_tokens,
        n_original_tokens,
        n_codebooks,
        start_audio_token_id,
        end_audio_token_id,
        device):
    # find start and end indices of audio tokens
    start = torch.nonzero(tokens == start_audio_token_id)
    end = torch.nonzero(tokens == end_audio_token_id)

    start = start[0, -1] + 1 if len(start) else 0
    end = end[0, -1] if len(end) else tokens.shape[-1]

    # subtract length of original vocabulary -> tokens in range [0, 1024)
    audio_tokens = tokens[start:end] % n_original_tokens
    reminder = audio_tokens.shape[-1] % n_codebooks

    if reminder:
        # pad if last frame is incomplete
        audio_tokens = torch.cat([audio_tokens, pad_tokens[reminder:]], dim=0)

    if n_codebooks > 1:
        transposed = audio_tokens.view(-1, n_codebooks).t()
    else:
        transposed = audio_tokens
    codes = transposed.view(n_codebooks, 1, -1).to(device)

    audio = quantizer.decode(codes).squeeze(0)

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return AudioSignal(audio.detach().cpu().numpy(), quantizer.sample_rate)


def prepare_librispeech(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("openslr/librispeech_asr", "clean", cache_dir=cache_dir)
    processed = raw.remove_columns(["chapter_id"])
    processed = processed.cast_column("speaker_id", Value("string"))
    return processed["train.100"], processed["validation"]


def prepare_tedlium(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("LIUM/tedlium", "release1", cache_dir=cache_dir)
    processed = raw.remove_columns(["gender"])
    return processed["train"], processed["validation"]


def prepare_parler_tts(cache_dir) -> tuple[Dataset, Dataset]:
    raw_mls = load_dataset("parler-tts/mls_eng", cache_dir=cache_dir)
    processed_mls = raw_mls.remove_columns(
        ["begin_time", "end_time", "speaker_id", "book_id", "audio_duration"]
    )
    processed_mls = processed_mls.rename_column("transcript", "text")

    return processed_mls["train"], processed_mls["dev"]


def prepare_synthetic(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("homebrewltd/instruction-speech-encodec-v1", cache_dir=cache_dir)
    processed = raw.remove_columns(["prompt", "length"])
    processed = processed.rename_column("answer", "text")
    splits = processed["train"].train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


def prepare_parler_tts_with_description(cache_dir) -> tuple[Dataset, Dataset]:
    audio = load_dataset("parler-tts/libritts_r_filtered", "clean", cache_dir=cache_dir)
    train_audio, val_audio = audio["train.clean.100"], audio["dev.clean"]

    columns = ["id", "text", "path", "text_description"]
    raw = load_dataset("parler-tts/libritts-r-filtered-speaker-descriptions", 'clean', cache_dir=cache_dir)
    processed = raw.remove_columns(list(set(raw.column_names["dev.clean"]) - set(columns)))
    train_text, val_text = processed["train.clean.100"], processed["dev.clean"]

    assert train_audio["id"] == train_text["id"] and val_audio["id"] == val_text["id"]

    audio_features_train = train_audio["audio"]
    audio_features_val = val_audio["audio"]
    train_text = train_text.map(lambda x, i: {"audio": audio_features_train[i]}, with_indices=True)
    val_text = val_text.map(lambda x, i: {"audio": audio_features_val[i]}, with_indices=True)
    return train_text, val_text


def get_last_checkpoint(save_dir):
    n_checkpoints = len(
        list(filter(lambda x: x.startswith("checkpoint"), os.listdir(save_dir)))
    )
    return n_checkpoints + 1


def save_checkpoint(
        model, accelerator, tokenizer, optimizer, scheduler, save_dir, checkpointing_steps
):
    accelerator.wait_for_everyone()
    state = model.state_dict()

    path = os.path.join(
        save_dir, f"checkpoint-{get_last_checkpoint(save_dir) * checkpointing_steps}"
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
