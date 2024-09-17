# Vikhr Salt: Speech And Language Transformer

![Vikhr Salt Logo](https://huggingface.co/Vikhrmodels/salt-116k/resolve/main/IMG_1304%20copy.png)

Vikhr Salt is a multimodal model based on a pre-trained large language model, extended with new audio tokens to handle both TTS (text-to-speech) and ASR (automatic speech recognition) tasks. The model incorporates two variants for encoding audio—Encodec and SpeechTokenizer—and achieves stable training by fine-tuning precision settings. This approach allows Vikhr Salt to leverage pre-existing LLM knowledge while effectively generating and understanding speech, marking a step forward in multimodal learning.

## Model  Authors 

Ksenya Sycheva, Konstantin Korolev, Aleksandr Nikolic
## Datasets 
- [TEDLIUM](https://huggingface.co/datasets/LIUM/tedlium)
- [Librispeech](https://huggingface.co/datasets/openslr/librispeech_asr)


## How to run

for single gpu
```
run_me_.sh

```

for multi gpu+ds2
```
sh run_me_ds2.sh

```
