#!/bin/bash

#SBATCH --job-name=g-base
#SBATCH -A irel
#SBATCH --output=runs/git/git-base.txt
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
EXP_NAME='base-test' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
# all-384,bbox-384,bbox-seg-384,ocr-384,desc-384,post-384,all-bboxseg-384,all-bboxseg-384
RUN_TYPE='train' # train,inference
DATE='6Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=1

if [ "$RUN_TYPE" = "train" ]; then
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=5
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    echo "Training model"

    # Exp 0 base test
     accelerate launch --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    --question_dir datasets/questions/all/master_sample_400.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 0 --lr 0

    # Exp 1 base
    # torchrun --standalone  --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900  \
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_sample_100.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 0 --lr 0

    # Exp 1 base + LR 
    # torchrun --standalone  --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900  \
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 0 --lr 1

    # Exp 1 base + DDP
     # Exp 1 base
    # accelerate config
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6  --train_dir datasets/$DATASET_SIZE \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_sample_400.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 
    # conda activate cqa-git
    # sbatch models-hf/ada-script-git-base-384.sh

    # Exp 2 wce
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/384a  \
    # --val_dir datasets/384a  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_sample_400.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 
    # conda activate cqa-wce
    # sbatch models-hf/ada-script-git-wce-384.sh

    # Exp 3 GIT - ocr + desc post + 576   
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets  \
    # --val_dir datasets  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1
    # conda activate cqa-git
    # sbatch models-hf/ada-script-git-post-384.sh


    # 4. GIT - ocr + desc post + 384 + wce - all 
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1

    # conda activate cqa-wce
    # sbatch models-hf/ada-script-git-all-384.sh


    # 5. GIT - 384 + bbxox
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_bbox.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --bbox 1
    # conda activate cqa-git
    # sbatch models-hf/ada-script-git-bbox-384.sh


    # # 6. GIT - 384 + bbox segment
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_bbox_segment.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --bbox_segment 1
    # conda activate cqa-git
    # sbatch models-hf/ada-script-git-bbox-seg-384.sh


    # # 7. GIT - 384 + ocr 
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_adv.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 0
    # # conda activate cqa-git
    # # sbatch models-hf/ada-script-git-ocr-3xw84.sh


    # # 8. GIT - 384 + desc
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_adv.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 0 --desc 1
    # # conda activate cqa-git
    # # sbatch models-hf/ada-script-git-desc-384.sh

    # # 9. GIT - 384 + ocr + desc (pre)
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_adv.json   --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1
    # # conda activate cqa-git
    # # sbatch models-hf/ada-script-git-pre-384.sh

    # # 10. GIT  - 384 + ocr + desc (post) + bbox_segment (all-bboxsegments) + wce 
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE  \
    # --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
    # --question_dir datasets/questions/all/master_bbox_segment_adv_ocr_400.json  --experiment_name ada-$MODEL-$EXP_NAME  \
    # --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1 --bbox_segment 1
    # # conda activate cqa-wce 
    # # sbatch models-hf/ada-script-git-all-bboxseg-384.sh

   

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    
    python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 2  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master.json \
    --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type 'dp'  --image_size 384  
 
    
  
else
    echo "Not valid"
fi 

