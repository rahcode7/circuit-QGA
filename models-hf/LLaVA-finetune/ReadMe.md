ssh rahul.mehta@ada
sinteractive -c 20 -g 1
conda activate cqa-base


#### Step 0 
```
pip install deepspeed
```

<!-- scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/LLaVA-finetune/ada-script-llava.sh rahul.mehta@ada:circuitQA/models-hf/
scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/LLaVA-finetune/train.py rahul.mehta@ada:circuitQA/LLaVA/llava/train/train.py


sbatch models-hf/ada-script-llava.sh 
cat runs/llava/llava-base.txt 
squeue -u $USER

 -->
#### Step 1 : Get instruction traiv/val datasets


```
pip install gdown==v4.6.0   # upgrade to latest version
gdown https://drive.google.com/drive/folders/1GMrFlCBH7utOaP8zKsqZFchabisu0qdE -O datasets-llava --folder

```

```
MODEL='llava'
CHECKPOINT="checkpoints-$MODEL-$EXP_NAME-$DATE"
mkdir $CHECKPOINT
export CUDA_VISIBLE_DEVICES=0

deepspeed LLaVA/llava/train/train_mem.py \
        --deepspeed LLaVA/scripts/zero2.json \
        --lora_enable True \
        --lora_r 128 \
        --lora_alpha 256 \
        --mm_projector_lr 2e-5 \
        --bits 4 \
        --model_name_or_path LLaVA/llava/llava-v1.5-7b \
        --version llava_llama_2 \
        --data_path datasets/llava/train/dataset.json \
        --validation_data_path datasets/llava/val/dataset.json \
        --image_folder datasets/384a/ \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $CHECKPOINT \
        --num_train_epochs 5 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy epoch --save_strategy steps \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb  
```