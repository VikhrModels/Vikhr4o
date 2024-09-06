import argparse

import torch
import yaml

from datasets import DatasetDict
from speechtokenizer import SpeechTokenizer
from transformers import AutoTokenizer

from src.data import load_data

parser = argparse.ArgumentParser(description="Train a model with configuration.")
parser.add_argument(
    "--config", type=str, help="Path to the config.yaml file", required=True
)
args = parser.parse_args()

# Load config
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

base_model = config["base_model"]
path_to_cache = config["path_to_cache"]

data = config["data"]
prepared_data_path = config["prepared_data_path"]

start_audio_token = config["start_audio_token"]
end_audio_token = config["end_audio_token"]
end_sequence_token = config["end_sequence_token"]
n_special_tokens = config["n_special_tokens"]

n_codebooks_tts = int(config["n_codebooks_tts"])
n_codebooks_asr = int(config["n_codebooks_asr"])
max_seq_length = int(config["max_seq_length"])
device = "cuda:0"


def quantize(row, quantizer):
    audio_data, sample_rate = row["audio_data"], row["sampling_rate"]

    audio = torch.tensor(audio_data).view(1, 1, audio_data.shape[-1]).float()
    audio = audio.to(device)
    codes = quantizer.encode(audio)
    codes = codes.squeeze(1)
    codes.cpu()

    del audio
    torch.cuda.empty_cache()

    return {"audio_tokens": codes.numpy()}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)

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

    train_dataset, val_dataset = load_data(data, tokenizer, path_to_cache)
    train_dataset = train_dataset.map(quantize, fn_kwargs={"quantizer": quantizer})
    val_dataset = val_dataset.map(quantize, fn_kwargs={"quantizer": quantizer})

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
        }
    )
    dataset.push_to_hub(
        prepared_data_path,
        storage_options={"format": "arrow"},
        private=True
    )

