#!/bin/bash

#SBATCH --job-name=p-ocrpost
#SBATCH -A research
#SBATCH --output=runs/pix/pix-ocr-post-infer.txt
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-post' 
RUN_TYPE='inference' # train,inference
DATE='31Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=1

if [ "$RUN_TYPE" = "train" ]; then
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1

    echo "Training model"

    # Exp 1 Base 384
    accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
        --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master_adv_ocr_sample_100.json  --experiment_name ada-$MODEL-$EXP_NAME \
        --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --ocr 1 

   
elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
   
    # Exp 1 Base 384
    python models-hf/$MODEL-vqa-inference.py --test_dir datasets/384a --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json   \
        --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type 'dp' --max_patches 512 --wce 1 --ocr 1
    
else
    echo "Not valid"
fi 

