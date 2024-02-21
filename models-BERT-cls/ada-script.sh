#!/bin/bash

#SBATCH --job-name=BERT-cls
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='BERT'
MACHINE_TYPE='dp' # ddp or dp or cpu
EXP_NAME='classifier' 
RUN_TYPE='train' # train,inference
DATE='24Nov'
CHECKPOINT="checkpoints-$MACHINE_TYPE-$EXP_NAME-$DATE"

if [ "$RUN_TYPE" = "train" ]; then
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    mkdir -p datasets/results/bert
    export NUM_NODES=1
    export EPOCHS=10
    export LOCAL_RANK=0

    # echo "Training model" 
    # base
    python models-$MODEL-cls/bert-model.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8 --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master.json  --experiment_name $MODEL_$EXP_NAME_$DATE \
        --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online 
  
elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    #rm -r datasets/results/$MODEL/$EXP_NAME

    # all
    # python models-hf/$MODEL-vqa-inference-adv.py --test_dir datasets/576a --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json \
    #         --results_dir datasets/results/$MODEL_$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 576  --ocr 1 --desc 1
                
else
    echo "Not valid"
fi 

