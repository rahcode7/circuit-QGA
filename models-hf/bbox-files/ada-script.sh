#!/bin/bash

#SBATCH --job-name=base-blip
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='blip'
MACHINE_TYPE='dp' # ddp or dp or cpu
EXP_NAME='bbox-384  ' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,ocr-post-desc,all,size-576
RUN_TYPE='train' # train,inference
DATE='25Nov'
CHECKPOINT="checkpoints-$MACHINE_TYPE-$EXP_NAME-$DATE"

if [ "$RUN_TYPE" = "train" ]; then
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0

    # echo "Training model" 
    # base
    python models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 2 --val_batch_size 2 --train_dir datasets/576a  \
        --val_dir datasets/576a  --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master.json  --experiment_name ada-blip-multi-$EXP_NAME \
        --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384

    # Distributed
    # torchrun --nnodes 1 --nproc_per_node 3 --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \
    # models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-blip-multi-$EXP_NAME \
    #     --ngpus 3 --machine_type $MACHINE_TYPE --wandb_status online 

    # TESTING 576 RESIZED IMAGES
    # python models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/576  \
    #     --val_dir datasets/576  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_400.json  --experiment_name ada-blip-multi-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384

    # TESTING 576 RESIZED IMAGES
    # python models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 1 --val_batch_size 1 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-blip-multi-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 
    
    #OCR and Description experiment
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 12 --val_batch_size 12 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME  \
    #     --ngpus 3 --machine_type $MACHINE_TYPE --wandb_status online --ocr 1 --desc 0 #--image_size 384 # for image size experiment 

    #OCR post processing experiment + desc
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr_sample_100.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --ocr 1 --desc 1

    # OCR post + desc + 576
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 12 --val_batch_size 12 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 3 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 576 --ocr 1 --desc 1

    # WCE + 576
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 2 --val_batch_size 2 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_100.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384

    # # OCR post + desc + wce
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --ocr 1 --desc 1

    # # wce + 576
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 12 --val_batch_size 12 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 3 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 576 

    # OCR post + desc + wce + 576
    # Change  for wce and modeling_blip.py for wce
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 12 --val_batch_size 12 --train_dir datasets  \
    #     --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 3 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 576 --ocr 1 --desc 1

    # Bbox
    python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/576a  \
        --val_dir datasets/576a  --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master_bbox.json  --experiment_name ada-$MODEL-$EXP_NAME \
        --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 384 --bbox 1 

    # Bbox segment
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/576a  \
    #     --val_dir datasets/576a  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox.json  --experiment_name ada-$MODEL-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 384 --bbox_segment 1 

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    #rm -r datasets/results/$MODEL/$EXP_NAME

    # Size,wce
    # python models-hf/blip-vqa-inference.py --test_dir datasets --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master.json \
    #         --results_dir datasets/results/$EXP_NAME --machine_type $MACHINE_TYPE

    # ocr and desc
    python models-hf/blip-vqa-inference.py --test_dir datasets/576a --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json \
            --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE --image_size 768 --ocr 0 --desc 0 

    # all
    # python models-hf/$MODEL-vqa-inference-adv.py --test_dir datasets/576a --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json \
    #         --results_dir datasets/results/$MODEL_$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 576  --ocr 1 --desc 1
                
else
    echo "Not valid"
fi 

