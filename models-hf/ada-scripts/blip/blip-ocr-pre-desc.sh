#!/bin/bash

#SBATCH --job-name=bi-opd-576
#SBATCH -A irel
#SBATCH --output=runs/blip/ocr-pre-desc-576-infer.txt
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='blip'
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-pre-desc' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,ocr-post-desc,all,size-576,all-384,post-384,bbox-seg-384,bbox-seg-384,all-384
# all-384,ocr-384,desc-384,post-384,all-bboxseg-384,pre-384,pre-384
RUN_TYPE='inference' # train,inference
DATE='10Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=1

if [ "$RUN_TYPE" = "train" ]; then
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    

    accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 2 --val_batch_size 2 --train_dir datasets/$DATASET_SIZE \
        --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master_bbox_and_segment_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
        --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 8 --image_size 576  --ocr 1 --desc 1 --bbox 0 --bbox_segment 0 --lr 1 


elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
   
    # Exp 1 . base 384
    export NUM_NODES=1
    # export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0

    python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_bbox_and_segment_adv.json \
    --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type 'dp'   --image_size 576  --ocr 1 --desc 1 --bbox 0 --bbox_segment 0 --lr 1 

else
    echo "Not valid"
fi 