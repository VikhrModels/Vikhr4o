
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class Vikhr4oDataset(Dataset):
    def __init__(self, dataset, tokenizer, device, asr=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.asr = asr
        self.device = device
        self.n_original_tokens = len(tokenizer) - 1024

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row["text"]
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        text_input_tokens = text_tokenized["input_ids"].to(self.device)

        return {
            "text_input_tokens": text_input_tokens,
            "audio_data": row["audio"]["array"],
            "sampling_rate": row["audio"]["sampling_rate"],
        }

