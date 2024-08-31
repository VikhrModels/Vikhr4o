CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
accelerate launch \
        --main_process_port 29001 \
        --config_file configs/accelerate_multigpu.yaml \
        train.py