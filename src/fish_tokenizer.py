import numpy as np

import torch
import torchaudio

from fish_speech.tokenizer import *
from tools.llama.generate import *
from tools.vqgan.inference import load_model as load_vqgan


device = "cuda"


def load_encoder(checkpoint_path, precision, is_agent=False):
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision)
    print(f"Restored model from checkpoint")

    decode_one_token = (
        decode_one_token_ar_agent if is_agent else decode_one_token_ar
    )
    print("Using DualARTransformer")

    return model.eval(), decode_one_token


class FishAudioTokenizer:
    def __init__(self, encoder_path: str, decoder_path: str,
                 decoder_config: str, tokenizer_path: str) -> None:
        self.tokenizer = FishTokenizer.from_pretrained(tokenizer_path)
        self.semantic_tokenizer, self.decoding_f = load_encoder(encoder_path,
                                                                torch.bfloat16,
                                                                )
        self.audio_tokenizer = load_vqgan(decoder_config, decoder_path)

        with torch.device(device):
            self.semantic_tokenizer.setup_caches(
                max_batch_size=1,
                max_seq_len=self.semantic_tokenizer.config.max_seq_len,
                dtype=next(self.semantic_tokenizer.parameters()).dtype,
            )

    def encode_text(self, text: str) -> list[int]:
        generator = generate_long(
            model=self.semantic_tokenizer,
            device=device,
            decode_one_token=self.decoding_f,
            text=text,
        )

        idx = 0
        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)

            elif response.action == "next":
                if codes:
                    np.save(f"codes_{idx}.npy", torch.cat(codes, dim=1).cpu().numpy())
                    print(f"Saved codes to codes_{idx}.npy")
                codes = []
                idx += 1
            else:
                print(f"Error: {response}")

        return codes

    def encode_audio(self, audios: torch.Tensor, sr: int) -> list[int]:
        if sr != self.audio_tokenizer.spec_transform.sample_rate:
            audios = torchaudio.functional.resample(
                audios, orig_freq=sr, new_freq=self.audio_tokenizer.spec_transform.sample_rate
            )

        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        tokens = self.audio_tokenizer.encode(audios, audio_lengths)[0][0]
        return tokens

    def decode(self, tokens: torch.Tensor, output_path: str) -> str:
        """tokens -> audio"""
        feature_lengths = torch.tensor([tokens.shape[1]], device=device)

        fake_audios, _ = self.audio_tokenizer.decode(
          indices=tokens[None], feature_lengths=feature_lengths
        )

        audio_time = fake_audios.shape[-1] / self.audio_tokenizer.spec_transform.sample_rate
        fake_audio = fake_audios[0].float().detach().cpu()
        torchaudio.save(output_path, fake_audio, self.audio_tokenizer.spec_transform.sample_rate)
