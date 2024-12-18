from datasets import load_dataset

import torch
from torch.utils.data import Dataset, ConcatDataset


class Vikhr4oDatasetBase(Dataset):
    def __init__(self, dataset, tokenizer, quantizer, asr: bool, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.quantizer = quantizer
        self.asr = asr

        if self.asr:
            self.n_codebooks = quantizer.asr_n_codebooks
        else:
            self.n_codebooks = quantizer.tts_n_codebooks

        self.max_seq_length = config["max_seq_length"]
        self.n_special_tokens = config["n_special_tokens"]

        self.soa = tokenizer(config["start_audio_token"], return_tensors="pt")[
                       "input_ids"
                   ][:, -1:]

        self.eoa = tokenizer(config["end_audio_token"], return_tensors="pt")[
                       "input_ids"
                   ][:, -1:]

        self.eos = tokenizer(config["end_sequence_token"], return_tensors="pt")[
                       "input_ids"
                   ][:, -1:]

    def __len__(self):
        return len(self.dataset)

    def get_text_tokens(self, row):
        text_tokenized = self.tokenizer(row["text"].lower(), return_tensors="pt")
        return text_tokenized["input_ids"]

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_input_tokens = self.get_text_tokens(row)

        if self.asr:
            audio_input_tokens = self.quantizer.quantize_asr(row)

            audio_length = audio_input_tokens.shape[-1]
            if audio_length > self.max_seq_length - 64:
                audio_length = self.max_seq_length // 6 * 5

            audio_length -= audio_length % self.n_codebooks
            text_length = min(
                self.max_seq_length - audio_length - self.n_special_tokens,
                text_input_tokens.shape[-1],
            )

        else:
            audio_input_tokens = self.quantizer.quantize_tts(row)

            text_length = text_input_tokens.shape[-1]
            if text_length > self.max_seq_length // 2:
                text_length = self.max_seq_length // 2

            audio_length = min(
                self.max_seq_length - text_length - self.n_special_tokens,
                audio_input_tokens.shape[-1],
            )
            audio_length -= audio_length % self.n_codebooks

        padding_size = (
                self.max_seq_length - text_length - audio_length - self.n_special_tokens
        )
        padding = torch.full((1, padding_size), self.tokenizer.pad_token_id, dtype=torch.int64)

        if self.asr:
            tokens = torch.cat(
                [
                    padding,
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    text_input_tokens.squeeze(1)[:, :text_length],
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)
        else:
            tokens = torch.cat(
                [
                    padding,
                    text_input_tokens.squeeze(1)[:, :text_length],
                    self.soa,
                    audio_input_tokens[:, :audio_length],
                    self.eoa,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)

        attention_mask = torch.cat(
            [torch.zeros(padding.shape), torch.ones((1, self.max_seq_length - padding_size))],
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
            text = "'{text}' is said with {voice_dsc}".format(
                text=row["text"], voice_dsc=row["text_description"]
            )
        else:
            text = "Say '{text}' with {voice_dsc}".format(
                text=row["text"], voice_dsc=row["text_description"]
            )
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        return text_tokenized["input_ids"]


def prepare_text_field(row):
    return {"text": row["json"]["text"]}


def load_tokenized_data(data_path: str):
    speech_path = data_path + "-speech"
    wav_path = data_path + "-wav-unify"

    speech = load_dataset(speech_path)
    train_speech, val_speech = speech["train"], speech["validation"]

    if "text" not in train_speech.column_names:
        train_speech = train_speech.map(prepare_text_field)
        val_speech = val_speech.map(prepare_text_field)

    wav = load_dataset(wav_path)
    train_wav, val_wav = wav["train"], wav["validation"]

    if "text" not in train_wav.column_names:
        train_wav = train_wav.map(prepare_text_field)
        val_wav = val_wav.map(prepare_text_field)

    train = train_speech.rename_column("audio_tokens", "audio_tokens_speech")
    train = train.add_column("audio_tokens_wav", train_wav["audio_tokens"])

    val = val_speech.rename_column("audio_tokens", "audio_tokens_speech")
    val = val.add_column("audio_tokens_wav", val_wav["audio_tokens"])

    return train, val


def load_train_val_splits(dataset: str, tokenizer, quantizer, config):
    train_ds, val_ds = load_tokenized_data(dataset)

    if "librispeech" in dataset or "emilia" in dataset:
        train_asr = Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, True, config)
        val_asr = Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, True, config)

        train_tts = Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, False, config)
        val_tts = Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, False, config)

        return [train_asr, train_tts], [val_asr, val_tts]

    elif "with_description" in dataset:
        train_asr = Vikhr4oDatasetVoiceDescription(
            train_ds, tokenizer, quantizer, True, config
        )
        val_asr = Vikhr4oDatasetVoiceDescription(
            val_ds, tokenizer, quantizer, True, config
        )

        train_tts = Vikhr4oDatasetVoiceDescription(
            train_ds, tokenizer, quantizer, False, config
        )
        val_tts = Vikhr4oDatasetVoiceDescription(
            val_ds, tokenizer, quantizer, False, config
        )

        return [train_asr, train_tts], [val_asr, val_tts]

    elif "homebrewltd" in dataset:
        train_asr = Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, True, config)
        val_asr = Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, True, config)

        return [train_asr], [val_asr]

    else:
        raise ValueError("Unknown dataset.")


def load_text_dataset(dataset_path: str, tokenizer, max_length: int):
    dataset = load_dataset(dataset_path)
    train, val = dataset["train"], dataset["validation"]
    template = "{instruction}\n{response}"

    train = train.map(
        lambda example: {
            "prompt": template.format(
                instruction=example["instruction"] + example["input"],
                response=example["output"],
            )
        }
    )

    val = val.map(
        lambda example: {
            "prompt": template.format(
                instruction=example["instruction"] + example["input"],
                response=example["output"],
            )
        }
    )

    train_tokenized = train.map(
        lambda examples: tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )

    val_tokenized = val.map(
        lambda examples: tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )

    train_tokenized = train_tokenized.map(lambda x: {"labels": x["input_ids"]})
    val_tokenized = val_tokenized.map(lambda x: {"labels": x["input_ids"]})

    train_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_tokenized, val_tokenized


def load_data(
        audio_datasets: list[str], tokenizer, quantizer, config
) -> tuple[Dataset, Dataset]:
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []

    for dataset in audio_datasets:
        train, val = load_train_val_splits(dataset, tokenizer, quantizer, config)
        train_datasets.extend(train)
        val_datasets.extend(val)

    if len(config["text_data"]):
        for text_ds in config["text_data"]:
            train, val = load_text_dataset(text_ds, tokenizer, config["max_seq_length"])
            train_datasets.append(train)
            val_datasets.append(val)

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
