CUDA_VISIBLE_DEVICES=0,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
accelerate launch \
        --config_file configs/accelerate/deepspeed.yaml \
        train.py --config configs/asr_tts/music.yaml