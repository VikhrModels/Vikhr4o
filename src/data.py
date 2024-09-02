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
    def __init__(self, dataset, tokenizer, asr: bool):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.asr = asr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row["text"]
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"]

        return {
            "text_input_tokens": text_input_tokens,
            "audio_data": row["audio"]["array"],
            "sampling_rate": row["audio"]["sampling_rate"],
            "asr": self.asr,
        }


class Vikhr4oDatasetVoiceDescription(Dataset):
    def __init__(self, dataset, tokenizer, asr: bool):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.asr = asr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if self.asr:
            text = "{text} is said with {voice_dsc}".format(text=row["text"], voice_dsc=row["text_description"])
        else:
            text = "Say {text} with {voice_dsc}".format(text=row["text"], voice_dsc=row["text_description"])
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"]

        return {
            "text_input_tokens": text_input_tokens,
            "audio_data": row["audio"]["array"],
            "sampling_rate": row["audio"]["sampling_rate"],
            "asr": self.asr,
        }


def load_data(datasets: list[str], tokenizer, cache_dir: str) -> tuple[Dataset, Dataset]:

    train_datasets = []
    val_datasets = []

    for dataset in datasets:
        train_ds, val_ds = DATASET_2_LOAD_FUNCTION[dataset](cache_dir)

        if "with_description" in dataset:
            train_tts = Vikhr4oDatasetVoiceDescription(train_ds, tokenizer, False)
            train_asr = Vikhr4oDatasetVoiceDescription(train_ds, tokenizer, True)

            val_tts = Vikhr4oDatasetVoiceDescription(val_ds, tokenizer, False)
            val_asr = Vikhr4oDatasetVoiceDescription(val_ds, tokenizer, True)

            train_datasets.extend([train_tts, val_tts])
        else:
            if "synthetic" not in dataset:
                train_tts = Vikhr4oDatasetBase(train_ds, tokenizer, False)
                val_tts = Vikhr4oDatasetBase(val_ds, tokenizer, False)

                train_datasets.extend([train_tts, val_tts])

            train_asr = Vikhr4oDatasetBase(train_ds, tokenizer, True)
            val_asr = Vikhr4oDatasetBase(val_ds, tokenizer, True)

        train_datasets.extend([train_asr])
        val_datasets.extend([val_asr])

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)
