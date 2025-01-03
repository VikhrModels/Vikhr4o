from typing import Optional
from audiotools import AudioSignal

import torch


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


def get_audio_start_end_tokens(
    tokens: torch.Tensor,
    start_audio_token_id: Optional[int],
    end_audio_token_id: Optional[int],
):
    # find start index of audio tokens
    if start_audio_token_id is not None:
        start = torch.nonzero(tokens == start_audio_token_id)
        start = start[0, -1] + 1 if len(start) else 0
    else:
        start = 0

    # find end index of audio tokens
    if end_audio_token_id is not None:
        end = torch.nonzero(tokens == end_audio_token_id)
        end = end[0, -1] if len(end) else tokens.shape[-1]
    else:
        end = tokens.shape[-1]

    assert (
        start < end
    ), f"Start of audio must be before end. Found: start - {start}, end - {end}"

    return start, end


def decode_audio_wav(
    tokens,
    quantizer,
    n_original_tokens,
    n_codebooks,
    start_audio_token_id: Optional[int] = None,
    end_audio_token_id: Optional[int] = None,
    device="cuda",
):
    # find audio start and end tokens
    start, end = get_audio_start_end_tokens(
        tokens, start_audio_token_id, end_audio_token_id
    )

    # subtract length of original vocabulary -> tokens in range [0, n_codebooks)
    audio_tokens = tokens[start:end] % n_original_tokens

    transposed = audio_tokens.view(-1, n_codebooks).t()
    codes = transposed.view(n_codebooks, 1, -1).to(device)

    features = quantizer.codes_to_features(codes)
    bandwidth_id = torch.tensor([0], device=device)

    audio = quantizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return AudioSignal(audio.detach().cpu().numpy(), 24000)


def decode_audio_speech(
    tokens,
    quantizer,
    n_original_tokens,
    n_codebooks,
    start_audio_token_id: Optional[int] = None,
    end_audio_token_id: Optional[int] = None,
    device="cuda",
):
    # find audio start and end tokens
    start, end = get_audio_start_end_tokens(
        tokens, start_audio_token_id, end_audio_token_id
    )

    # substract length of original vocabulary -> tokens in range [0, 1024)
    audio_tokens = tokens[start:end] % n_original_tokens
    remainder = audio_tokens.shape[-1] % n_codebooks

    if remainder:
        # pad if last frame is incomplete
        # zero padding is used now, for speechtokenizer using get_audio_padding_tokens is also possible
        pad_tokens = torch.zeros(
            1, n_codebooks - remainder, device="cuda", dtype=torch.long
        )
        audio_tokens = torch.cat([audio_tokens, pad_tokens], dim=0)

    transposed = audio_tokens.view(-1, n_codebooks).t()
    codes = transposed.view(n_codebooks, 1, -1).to(device)

    audio = quantizer.decode(codes).squeeze(0)

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return AudioSignal(audio.detach().cpu().numpy(), quantizer.sample_rate)


def decode_audio_fish(
    tokens,
    quantizer,
    n_original_tokens,
    n_codebooks,
    start_audio_token_id: Optional[int] = None,
    end_audio_token_id: Optional[int] = None,
    device="cuda",
):
    # find audio start and end tokens
    start, end = get_audio_start_end_tokens(
        tokens, start_audio_token_id, end_audio_token_id
    )

    # subtract length of original vocabulary -> tokens in range [0, 1024)
    audio_tokens = tokens[start:end] % n_original_tokens
    remainder = audio_tokens.shape[-1] % n_codebooks

    if remainder:
        # pad if last frame is incomplete, zero padding is used
        pad_tokens = torch.zeros(
            (1, n_codebooks - remainder), device="cuda", dtype=torch.long
        )
        audio_tokens = torch.cat([audio_tokens, pad_tokens], dim=0)

    transposed = audio_tokens.view(-1, n_codebooks).t()
    codes = transposed.to(device)

    audio = quantizer.decode(codes).squeeze(0)

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return AudioSignal(audio.detach().cpu().numpy(), quantizer.sample_rate)
