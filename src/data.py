from datasets import load_dataset

import torch
from torch.utils.data import Dataset, ConcatDataset

from src.utils import prepare_librispeech, prepare_parler_tts, prepare_synthetic, prepare_tedlium, \
    prepare_parler_tts_with_description

DATASET_2_LOAD_FUNCTION = {
    "librispeech": prepare_librispeech,
    "parler-tts": prepare_parler_tts,
    "parler_tts_with_description": prepare_parler_tts_with_description,
    "synthetic": prepare_synthetic,
    "tedlium": prepare_tedlium,
}


class Vikhr4oDatasetBase(Dataset):
    def __init__(self, dataset, tokenizer, asr: bool, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.asr = asr

        if asr:
            self.n_codebooks = config["n_codebooks_asr"]
        else:
            self.n_codebooks = config["n_codebooks_tts"]

        self.max_seq_length = config["max_seq_length"]
        self.n_special_tokens = config["n_special_tokens"]

        self.soa = tokenizer(config["start_audio_token"], return_tensors="pt")["input_ids"][
                   :, -1:
                   ]
        self.eoa = tokenizer(config["end_audio_token"], return_tensors="pt")["input_ids"][
                   :, -1:
                   ]
        self.eos = tokenizer(config["end_sequence_token"], return_tensors="pt")["input_ids"][
                   :, -1:
                   ]

    def __len__(self):
        return len(self.dataset)

    def get_text_tokens(self, row):
        text_tokenized = self.tokenizer(row["text"], return_tensors="pt")
        return text_tokenized["input_ids"]

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_input_tokens = self.get_text_tokens(row)

        codes = torch.tensor(row["audio_tokens"])
        raw_audio_tokens = codes[:self.n_codebooks]

        audio_input_tokens = raw_audio_tokens.t().contiguous().view(1, -1)
        audio_length = min(
            self.max_seq_length - text_input_tokens.shape[-1] - self.n_special_tokens,
            audio_input_tokens.shape[-1],
        )
        audio_length -= audio_length % self.n_codebooks
        padding_size = (
                self.max_seq_length
                - text_input_tokens.shape[-1]
                - audio_length
                - self.n_special_tokens
        )
        padding = torch.zeros((1, padding_size), dtype=torch.int64)

        if self.asr:
            tokens = torch.cat(
                [
                    padding,
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    text_input_tokens.squeeze(1),
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)
        else:
            tokens = torch.cat(
                [
                    padding,
                    text_input_tokens.squeeze(1),
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)

        attention_mask = torch.cat(
            [padding, torch.ones((1, self.max_seq_length - padding_size))],
            dim=1,
        ).squeeze(0)

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": tokens.clone(),
        }


class Vikhr4oDatasetVoiceDescription(Vikhr4oDatasetBase):
    def get_text_tokens(self, row):
        if self.asr:
            text = "{text} is said with {voice_dsc}".format(text=row["text"], voice_dsc=row["text_description"])
        else:
            text = "Say {text} with {voice_dsc}".format(text=row["text"], voice_dsc=row["text_description"])
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        return text_tokenized["input_ids"]


def load_data(datasets: list[str], tokenizer, config) -> tuple[Dataset, Dataset]:
    train_datasets = []
    val_datasets = []

    for dataset in datasets:
        ds = load_dataset(dataset)
        train_ds, val_ds = ds["train"], ds["validation"]

        if "with_description" in dataset:
            train_tts = Vikhr4oDatasetVoiceDescription(train_ds, tokenizer, False, config)
            train_asr = Vikhr4oDatasetVoiceDescription(train_ds, tokenizer, True, config)

            val_tts = Vikhr4oDatasetVoiceDescription(val_ds, tokenizer, False, config)
            val_asr = Vikhr4oDatasetVoiceDescription(val_ds, tokenizer, True, config)

            train_datasets.append(train_tts)
            val_datasets.append(val_tts)
        else:
            if "synthetic" not in dataset:
                train_tts = Vikhr4oDatasetBase(train_ds, tokenizer, False, config)
                val_tts = Vikhr4oDatasetBase(val_ds, tokenizer, False, config)

                train_datasets.append(train_tts)
                val_datasets.append(val_tts)

            train_asr = Vikhr4oDatasetBase(train_ds, tokenizer, True, config)
            val_asr = Vikhr4oDatasetBase(val_ds, tokenizer, True, config)

        train_datasets.extend([train_asr])
        val_datasets.extend([val_asr])

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
