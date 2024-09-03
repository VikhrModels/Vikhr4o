CUDA_VISIBLE_DEVICES=0,3 NCCL_IB_GID_INDEX=3 NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
accelerate launch \
        --config_file configs/accelerate_multigpu.yaml \
        train.py --config configs/asr_tts.yaml
