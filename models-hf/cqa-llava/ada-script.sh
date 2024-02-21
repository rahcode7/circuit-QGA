#!/bin/bash

#SBATCH --job-name=llava-base
#SBATCH -A research
#SBATCH --output=runs/llava/llava-base.txt
#SBATCH -n 10
#SBATCH --gres=gpu:4
#SBATCH --mem=30G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='llava'  # git,blip
 # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='inference' # train,inference
DATE='13Feb'
DATASET_SIZE='384a'
NUM_GPUS=4

if [ "$RUN_TYPE" = "inference" ]; then
    export NUM_NODES=1
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    echo "Infer model"

    # Exp 1 Base 384
    #accelerate launch --multi_gpu --num_processes=$NUM_GPUS 
    
    # Run local
    # python LLaVA/LLaVa-mac/cqa-llava/eval-single.py --question_dir datasets/questions/all/master.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME
    
    # BBox-segments 
    # EXP_NAME='bbox_segment'
    # mkdir models-LLAVA-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python LLaVA/LLaVa-mac/cqa-llava/eval-single.py --question_dir datasets/questions/all/master_bbox_segment.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-LLAVA-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # Desc
    EXP_NAME='desc'
    mkdir models-LLAVA-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    python LLaVA/LLaVa-mac/cqa-llava/eval-single.py --question_dir datasets/questions/all/master_adv.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir models-LLAVA-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # Ada
    #accelerate launch --multi_gpu --num_processes=$NUM_GPUS  models-hf/LLaVA/cqa-llava/eval-single.py --question_dir datasets/questions/all/master.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME

    EXP_NAME='desc'
    # # mkdir datasets/results/llava/$DATASET_SIZE/
    mkdir datasets/results/llava/$DATASET_SIZE/$EXP_NAME
    #accelerate launch --multi_gpu --num_processes=$NUM_GPUS  
    accelerate launch --multi_gpu --num_processes=$NUM_GPUS  LLaVA/cqa-llava/eval-single.py --question_dir datasets/questions/all/master_adv.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/$MODEL/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME


else
    echo "Not valid"
fi 

