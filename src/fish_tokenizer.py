from typing import Optional

import torch

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
    def __init__(
        self,
        decoder_path: str,
        decoder_config: str,
        encoder_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None
    ) -> None:

        self.audio_tokenizer = load_vqgan(decoder_config, decoder_path)
        self.sample_rate = self.audio_tokenizer.spec_transform.sample_rate

        if tokenizer_path is not None:
            self.tokenizer = FishTokenizer.from_pretrained(tokenizer_path)

        if encoder_path is not None:
            self.semantic_tokenizer, self.decoding_f = load_encoder(
                encoder_path, torch.bfloat16,
            )

            with torch.device(device):
                self.semantic_tokenizer.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.semantic_tokenizer.config.max_seq_len,
                    dtype=next(self.semantic_tokenizer.parameters()).dtype,
                )

    @property
    def semantic_codebook_size(self):
        return self.semantic_tokenizer.config.codebook_size

    @property
    def semantic_num_codebooks(self):
        return self.semantic_tokenizer.config.num_codebooks

    @property
    def codebook_size(self):
        return self.audio_tokenizer.quantizer.residual_fsq.codebook_size

    @property
    def num_codebooks(self):
        return len(self.audio_tokenizer.quantizer.residual_fsq.rvqs)

    def encode_text(self, text: str) -> torch.Tensor:
        generator = generate_long(
            model=self.semantic_tokenizer,
            device=device,
            decode_one_token=self.decoding_f,
            text=text,
        )

        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)

            elif response.action == "next":
                if codes:
                    codes = torch.cat(codes, dim=1)
                break
            else:
                print(f"Error: {response}")

        return codes

    def encode_audio(self, audios: torch.Tensor) -> torch.Tensor:
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        tokens = self.audio_tokenizer.encode(audios, audio_lengths)[0][0]
        return tokens

    def decode(self, tokens: torch.Tensor) -> str:
        """tokens -> audio"""
        feature_lengths = torch.tensor([tokens.shape[1]], device=device)

        fake_audios, _ = self.audio_tokenizer.decode(
          indices=tokens[None], feature_lengths=feature_lengths
        )

        fake_audio = fake_audios[0].float().detach().cpu()
        return fake_audio
