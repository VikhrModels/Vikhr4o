CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
accelerate launch \
        --main_process_port 29001 \
        --config_file configs/accelerate.yaml \
        train.py --config /configs/asr_tts.yaml