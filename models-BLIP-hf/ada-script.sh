#!/bin/bash

#SBATCH --job-name=VQA-infer-wce
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1


MACHINE_TYPE='dp' # ddp or dp 
EXP_NAME='size' # wce,size,focal,desc
RUN_TYPE='train' # train,inference
DATE='22Aug'
CHECKPOINT="checkpoints-$MACHINE_TYPE-$EXP_NAME-$DATE"

if [ "$RUN_TYPE" = "train" ]; then
    mkdir $CHECKPOINT
    export EPOCHS=2
    
    if [ "$MACHINE_TYPE" = "dp" ]; then 
        echo "Training model"
        python models-BLIP-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets  \
            --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
            --question_dir datasets/questions/all/master_sample_100.json  --experiment_name ada-blip-multi-$EXP_NAME --is_distributed False \
            --ngpus 2 --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 # 480,576,784 for image size experiment,defaults to 384 
           
    elif [ "$MACHINE_TYPE" = "ddp" ]; then
        export NUM_NODES=1
        export LOCAL_RANK=0
        export NUM_GPUS_PER_NODE=2
        export OMP_NUM_THREADS=2

        torchrun \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --nnodes=$NUM_NODES models-BLIP-hf/blip-vqa-train-$MACHINE_TYPE-acc.py --num_epochs $EPOCHS --train_batch_size 1 --val_batch_size 1 --train_dir datasets  \
            --val_dir datasets  --checkpoint_dir  $CHECKPOINT  \
            --question_dir datasets/questions/all/master.json  --experiment_name ada-blip-multi-$EXP_NAME --is_distributed False \
            --ngpus $NUM_GPUS_PER_NODE --machine_type $MACHINE_TYPE --wandb_status disabled #--image_size 384 # for image size experiment 

    else
        echo "Not valid"
    fi

elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    mkdir datasets/results/$EXP_NAME
    python models-BLIP-hf/blip-vqa-inference.py --test_dir datasets --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master.json \
        --results_dir datasets/results/$EXP_NAME --machine_type $MACHINE_TYPE
else
    echo "Not valid"
fi 

