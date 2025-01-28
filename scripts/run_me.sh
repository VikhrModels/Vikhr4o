PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 TOKENIZERS_PARALLELISM=false \
accelerate launch \
    --main_process_port 29501 \
    --config_file configs/accelerate/accelerate.yaml \
    train.py --config configs/asr_tts/music.yaml
