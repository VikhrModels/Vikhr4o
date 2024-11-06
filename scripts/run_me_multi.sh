TOKENIZERS_PARALLELISM=false NCCL_IB_GID_INDEX=3 NCCL_P2P_DISABLE=1 NCCL_P2P_LEVEL=NVL accelerate launch \
        --config_file configs/accelerate_multigpu.yaml \
        --main_process_port 29000 \
        train.py --config configs/asr_tts.yaml
