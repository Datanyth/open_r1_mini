export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN=<your_huggingface_token>
accelerate launch src/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name presencesw/en_processed_open-s1 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 20 \
    --max_seq_length 4096 \
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
    # --dataset_text_field "conversatins" \
    # --use_liger_kernel \
    # --dataset_name open-r1/OpenR1-Math-220k \
