import argparse
import yaml

import librosa
import numpy as np
import torch

from datasets import DatasetDict
from speechtokenizer import SpeechTokenizer

from src.utils import DATASET_2_LOAD_FUNCTION


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
    audio_data, sample_rate = row["audio"]["array"], row["audio"]["sampling_rate"]
    if sample_rate != quantizer.sample_rate:
        audio_data = librosa.resample(
            np.array(audio_data), orig_sr=sample_rate, target_sr=quantizer.sample_rate
        )

    audio = torch.tensor(audio_data).view(1, 1, len(audio_data)).float()
    audio = audio.to(device)
    codes = quantizer.encode(audio)
    codes = codes.squeeze(1)
    codes = codes.cpu()

    del audio
    torch.cuda.empty_cache()

    return {"audio_tokens": codes.numpy()}


if __name__ == "__main__":
    config_path = config["quantizer_config_path"]
    ckpt_path = config["quantizer_ckpt_path"]
    quantizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    quantizer.eval().to(device)

    codebook_size = quantizer.quantizer.bins

    train_dataset, val_dataset = DATASET_2_LOAD_FUNCTION[data[0]](path_to_cache)

    train_dataset = train_dataset.map(
        quantize,
        fn_kwargs={"quantizer": quantizer},
        cache_file_name="cache/tokenize_train_homebrew",
    )
    val_dataset = val_dataset.map(
        quantize,
        fn_kwargs={"quantizer": quantizer},
        cache_file_name="cache/tokenize_val_homebrew",
    )

    train_dataset = train_dataset.remove_columns(["audio"])
    val_dataset = val_dataset.remove_columns(["audio"])

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
        }
    )
    dataset.push_to_hub(prepared_data_path, private=True)
