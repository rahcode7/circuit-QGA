#!/bin/bash

#SBATCH --job-name=ip-bbox
#SBATCH -A irel
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

export MODEL='pix'
export MACHINE_TYPE='dp' # ddp or dp or cpu
export EXP_NAME='bbox-384' # base-384,wce-384,desc-384,ocr-384,post-384,bbox-384,bbox-seg-384
export RUN_TYPE='inference' # train,inference
export DATE='4Dec'
export CHECKPOINT="checkpoints-$MODEL-$EXP_NAME-$DATE"

if [ "$RUN_TYPE" = "train" ]; then
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0

    echo "Training model"

    # Exp 1 Base 384
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512
    # conda activate cqa-size
    # sbatch models-hf/pix-base-384.sh

    # Exp 2 WCE 384
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1  --machine_type $MACHINE_TYPE --wandb_status online  --max_patches 512
    # # conda activate cqa-wce
    # sbatch models-hf/pix-wce-384.sh
    
    # Exp 3 OCR post + desc + 576
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512  --ocr 1 --desc 1 
    # conda activate cqa-size
    # sbatch models-hf/pix-post-384.sh

    # Exp 4. OCR post + desc + wce - all
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --ocr 1 --desc 1
    # conda activate cqa-wce 
    # sbatch models-hf/pix-all-384.sh

    # Exp 5. BBox
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --max_patches 512 --bbox 1
    # conda activate cqa-size
    # sbatch models-hf/pix-bbox-384.sh

    # Exp 6. BBox segment
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox_segment.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --max_patches 512 --bbox_segment 1
    # # conda activate cqa-size
    # sbatch models-hf/pix-bbox-seg-384.sh

    # Exp 7 OCR
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a  \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME  \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --ocr 1 --desc 0 
    # # conda activate cqa-size
    # sbatch models-hf/pix-ocr-384.sh 

    # Exp 8 DESC
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a  \
    #     --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME  \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --ocr 0 --desc 1
    # conda activate cqa-size
    # sbatch models-hf/pix-desc-384.sh

    # Exp 9. 384 + ocr + desc (pre)
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a  \
    # --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    # --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --max_patches 512 --ocr 1 --desc 1
    # conda activate cqa-size
    # sbatch models-hf/pix-pre-384.sh

    # Exp 10. BLIP - 384 + ocr + desc (post) + bbox_segment (all-bboxsegments) + wce
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/384a  \
    # --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  \
    # --question_dir datasets/questions/all/master_bbox_segment_adv_ocr.json  --max_patches 512 --experiment_name ada-$MODEL-$EXP_NAME \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --ocr 1 --desc 1 
    # conda activate cqa-wce 
    # sbatch models-hf/pix-all-bboxseg-384.sh

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    # rm -r datasets/results/$EXP_NAME
   

    
else
    echo "Not valid"
fi 

