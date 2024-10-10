CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
accelerate launch \
    --config_file configs/accelerate.yaml \
    train.py --config configs/asr_tts.yaml