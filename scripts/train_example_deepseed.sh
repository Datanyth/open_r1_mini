weight_path=/host/ssd/hf_models/llama2-7b-hf
# weight_path=/host/ssd/hf_models/Meta-Llama-3.1-8B
export WANDB_MODE=disabled
num_gpus=2
epoch=3
mbs=2
MODE=${1:-zero1tp} 
if [ "$MODE" == "zero1tp" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=2
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
elif [ "$MODE" == "zero2tp" ]; then
  ZERO_STAGE=2
  AUTOTP_SIZE=2
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
elif [ "$MODE" == "zero1" ]; then
  ZERO_STAGE=1
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "zero2" ]; then
  ZERO_STAGE=2
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "zero3" ]; then
  ZERO_STAGE=3
  AUTOTP_SIZE=0
  per_device_train_batch_size=$mbs
elif [ "$MODE" == "tp" ]; then
  ZERO_STAGE=0
  AUTOTP_SIZE=2
  per_device_train_batch_size=$((mbs * AUTOTP_SIZE))
else
  echo "error '$MODE',please use 'zero' or 'tp'。"
  exit 1
fi
TEMPLATE_FILE="configs/ds_config_temp.json"
OUTPUT_FILE="configs/ds_config.json"
sed -e "s/\${zero_stage}/${ZERO_STAGE}/g" \
    -e "s/\${autotp_size}/${AUTOTP_SIZE}/g" \
    $TEMPLATE_FILE > $OUTPUT_FILE


# export HF_TOKEN=<your_huggingface_token>
deepspeed --num_gpus $num_gpus --master_port 51336 src/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name presencesw/en_processed_open-s1 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 20 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-SFT-1 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name Qwen2.5-1.5B-Open-R1-SFT-1 \
    --optim "adamw_8bit" \
    --overwrite_output_dir True \
    --hub_model_id presencesw/Qwen2.5-1.5B-Open-R1-SFT-1 \
    --push_to_hub True \
    --deepspeed "./deepspeed_config/ds_config.json"
    # --dataset_text_field "conversatins" \
    # --use_liger_kernel \
    # --dataset_name open-r1/OpenR1-Math-220k \
