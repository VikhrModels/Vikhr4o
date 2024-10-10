import torch


def get_start_tokens(quantizer_config, n_base_tokens):
    asr = quantizer_config["asr"]
    tts = quantizer_config["tts"]

    asr_types = [x["quantizer"] for x in asr]
    tts_types = [x["quantizer"] for x in tts]
    types = asr_types + tts_types

    tokens_config = {}
    if "speech" in types:
        tokens_config["speech"] = n_base_tokens
        n_base_tokens += quantizer_config["speech"]["n_new_tokens"]
    if "wav" in types:
        tokens_config["speech"] = n_base_tokens
        n_base_tokens += quantizer_config["wav"]["n_new_tokens"]

    return tokens_config


class AudioTokenizer:
    def __init__(self, quantizer_config, tokens_config):
        self.asr_config = quantizer_config["asr"]
        self.tts_config = quantizer_config["tts"]
        self.tokens_config = tokens_config

        self.asr_n_codebooks = max([x["n_codebooks"] for x in quantizer_config["asr"]])
        self.tts_n_codebooks = max([x["n_codebooks"] for x in quantizer_config["tts"]])

    def quantize(self, row, quantizer):
        if quantizer["quantizer"] == "speech":
            raw_tokens = torch.tensor(row["audio_tokens_speech"])
            audio_tokens = raw_tokens[:quantizer["n_codebooks"]] + self.tokens_config["speech"]
        elif quantizer["quantizer"] == "wav":
            raw_tokens = torch.tensor(row["audio_tokens_wav"])
            audio_tokens = raw_tokens[:quantizer["n_codebooks"]] + self.tokens_config["wav"]
        else:
            raise ValueError("Unknown quantizer.")

        return audio_tokens.t().contiguous().view(1, -1)

    def quantize_asr(self, row):
        codes = []

        for quantizer in self.asr_config:
            audio_tokens = self.quantize(row, quantizer)
            codes.append(audio_tokens)

        return torch.cat(codes, dim=0)

    def quantize_tts(self, row):
        codes = []

        for quantizer in self.tts_config:
            audio_tokens = self.quantize(row, quantizer)
            codes.append(audio_tokens)

        return torch.cat(codes, dim=0)
