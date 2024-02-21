#!/bin/bash
#SBATCH --job-name=Iblip
#SBATCH -A research
#SBATCH --output=runs/Iblip/iblip-base.txt
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='InstructBLIP'  # git,blip
EXP_NAME='base' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='inference' # train,inference
DATE='21Feb'
DATASET_SIZE='384a'
NUM_GPUS=1

if [ "$RUN_TYPE" = "inference" ]; then
    export NUM_NODES=1
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0

    echo "Infer model"

    # Exp 1 Base 384
    #accelerate launch --multi_gpu --num_processes=$NUM_GPUS 
    
   
    # OCR-post

    # EXP_NAME='ocr-post'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_adv_ocr.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # EXP_NAME='bbox-segment'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_bbox_segment.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    MODEL='InstructBLIP'
    EXP_NAME='base'

    mkdir datasets/results/InstructBLIP

    python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME


    # MODEL='InstructBLIP'
    # EXP_NAME='desc'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_adv.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # MODEL='InstructBLIP'
    # EXP_NAME='bbox'
    # DATASET_SIZE='384a'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # MODEL='InstructBLIP'
    # EXP_NAME='ocr-pre'
    # DATASET_SIZE='384a'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_adv.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # MODEL='InstructBLIP'
    # EXP_NAME='bbox-yolo'
    # DATASET_SIZE='384a'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_bbox_yolo.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME

    # MODEL='InstructBLIP'
    # EXP_NAME='bbox-segment-yolo'
    # DATASET_SIZE='384a'
    # mkdir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME
    # python models-InstructBLIP-hf/iblip-eval-single.py  --question_dir datasets/questions/all/master_bbox_segment_yolo.json  \
    # --image_dir datasets/$DATASET_SIZE --results_dir models-InstructBLIP-hf/results-ddp/$DATASET_SIZE/$EXP_NAME --exp_name $EXP_NAME


else
    echo "Not valid"
fi 

