# Vikhr Salt: Speech And Language Transformer

![Vikhr Salt Logo](https://huggingface.co/Vikhrmodels/salt-116k/resolve/main/IMG_1304%20copy.png)

Vikhr Salt is a multimodal model based on a pre-trained large language model, extended with new audio tokens to handle both TTS (text-to-speech) and ASR (automatic speech recognition) tasks. The model incorporates two variants for encoding audio—Encodec and SpeechTokenizer—and achieves stable training by fine-tuning precision settings. This approach allows Vikhr Salt to leverage pre-existing LLM knowledge while effectively generating and understanding speech, marking a step forward in multimodal learning.

## Model  Authors 

Ksenia Sycheva, Konstantin Korolev, Aleksandr Nikolic
## Datasets 
- [TEDLIUM](https://huggingface.co/datasets/LIUM/tedlium)
- [Librispeech](https://huggingface.co/datasets/openslr/librispeech_asr)


## How to run
### Preparing Data
To tokenize data run [prepare_data.py](prepare_data.py). Configs for different tokenizers ([SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer), [WavTokenizer](https://github.com/jishengpeng/WavTokenizer/), [FishTokenizer](https://github.com/fishaudio/fish-speech/)) are available in [this](configs/quantization) folder. 
```
python prepare_data.py --config configs/quantization/<your-tokenizer-config>.yaml

```

### Training
It is possible to configure tokenization for TTS and ASR differently:
- different number of tokens 
- different tokenizers 

To do that specify type of quantizer and number of codebooks for both tasks. Examples of configs can be found [here](configs/asr_tts).
Notes:
1. music/other non-speech data is only supported by [this](configs/quantization/quantization-wav-music.yaml) version of WavTokenizer
2. WavTokenizer has fixed number of codebooks = 1, for SpeechTokenizer values between 1 and 8 can be chosen 

for single gpu
```
source scripts/run_me.sh

```

for multi gpu+ds2
```
source scripts/run_me_ds2.sh

```
