#!/bin/bash

#SBATCH --job-name=gi-desc
#SBATCH -A research
#SBATCH --output=runs/git/git-desc-384-infer.txt
#SBATCH -n 10   
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

export CUDA_LAUNCH_BLOCKING=1

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='desc' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
# all-384,bbox-384,bbox-seg-384,ocr-384,desc-384,post-384,all-bboxseg-384,all-bboxseg-384
RUN_TYPE='inference' # train,inference
DATE='10Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=1

if [ "$RUN_TYPE" = "train" ]; then
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    echo "Training model"

    # OCR 
    # torchrun --standalone  --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900  \
    accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    --question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name adalocal-$MODEL-$EXP_NAME  \
    --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --lr 1 --desc 1

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    
   python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 1  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json \
    --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type 'dp'  --image_size 384   --lr 1 --desc 1
   

else
    echo "Not valid"
fi 

