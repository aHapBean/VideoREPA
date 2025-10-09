#!/usr/bin/env bash
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
source /cpfs/user/zhangxiangdong/miniconda3/bin/activate
conda activate videorepa

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

BASE_PATH="/path/to/VideoREPA"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    # training data
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "${BASE_PATH}/finetune"
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16 49x480x720 81x768x1360
    # place holder (useless)
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 4
    --seed 42 

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 4
    --gradient_accumulation_steps 1 # https://www.deepspeed.ai/docs/config-json/
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 800 # save checkpoint every x steps NOTE here
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "${BASE_PATH}/finetune/validation"
    --validation_steps 800  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

# Model Configuration
MODEL_ARGS=(
    --model_path "${BASE_PATH}/ckpt/cogvideox-2b"
    --model_name "cogvideox-t2v-align"  # ["cogvideox-t2v", "cogvideo-t2v-align"] -> cogvideo-t2v-align means representation alignment enabled
    --model_type "t2v"
    --training_type "sft"   # lora, sft
    # enable this only when caching
    # --precomputing
)

# VideoREPA Configuration
VideoREPA_ARGS=(
    --loss token_relation_distillation    # token_relation_distillation (VideoREPA), cosine_similarity (REPA)
    --align_models VideoMAEv2   # DINOv2 VideoMAEv2 VideoMAE OminiMAE VJEPA VJEPA2
    --align_layer 18            # depth in denoise transformer for alignment
    --align_dims 768            # feature dimensions in vision encoder
    --proj_coeff 0.5            # loss coefficient 0.5 by default for cosine loss
    --margin 0.1                # the margin for TRD loss in VideoREPA (may be different for various VFMs)
    --comment 'VideoREPA_2B'    # run_name comment for readable wandb log 
    --learning_rate 2e-6
)

JOB_NAME='test'
GPUS=${GPUS:-1}        
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# Combine all arguments and launch training
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${VideoREPA_ARGS[@]}"
