#!/bin/bash

#SBATCH --job-name=blip-all20
#SBATCH -A research
#SBATCH --output=runs/blip/blip-all.txt
#SBATCH -n 10
#SBATCH --gres=gpu:3
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=rahul.mehta@research.iiit.ac.in
#SBATCH -N 1

MODEL='blip'
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='all' # wce,size,focal,desc,ocr,ocr-post,ocr-pre,size-768,ocr-post-desc,all,size-576,all-384,post-384,bbox-seg-384,bbox-seg-384,all-384
# all-384,ocr-384,desc-384,post-384,all-bboxseg-384,pre-384,pre-384
RUN_TYPE='train' # train,inference
DATE='30Dec'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=3

if [ "$RUN_TYPE" = "train" ]; then
    rm -rf $CHECKPOINT
    mkdir $CHECKPOINT
    export NUM_NODES=1
    export EPOCHS=20
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0,1,2
    
    # --num_processes=$NUM_GPUS

    # echo "Training model" 
    # base
    # python models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_100.json  --experiment_name ada-blip-multi-$EXP_NAME \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384

    # Distributed base
    # torchrun --nnodes 1 --nproc_per_node 3 --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \
    # models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4

    # Test adamw + dp 
    # torchrun --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \
    # models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_400.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online # --accumulation_steps 1

    # base full + 8 bit adam
    # torchrun --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \
    # models-hf/blip-vqa-train-$MACHINE_TYPE-adamw.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_400.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4
  
    # Accelerate launch
    #accelerate config 
    #accelerate launch  --num_machines $NUM_GPUS 
    #accelerate launch --multi_gpu 
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 # --num_machines=$NUM_GPUS 


    # Exp 1. base
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 2 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384

    ### Exp wce

    # accelerate launch --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --wce 1


    # Exp 2 . base + ocr 
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 6 --val_batch_size 6 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --ocr 1 

#     # Exp 3. base + desc
#    accelerate launch --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
#         --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
#         --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
#         --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --desc 


#     # Exp 4. base + bbox
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --bbox 1

#     # Exp 4. base + bboxsegment
    # accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox_segment.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --bbox_segment 1

#     ## Exp 5. all
    accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
        --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
        --question_dir datasets/questions/all/master_bbox_and_segment_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
        --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --ocr 1 --desc 1 --bbox 1 --bbox_segment 1 

#     accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 15 --val_batch_size 15 --train_dir datasets/$DATASET_SIZE \
#         --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
#         --question_dir datasets/questions/all/master_bbox_segment_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
#         --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4  --ocr 1 --desc 1 --bbox 1 --bbox_segment 1


    # # Local
    # mkdir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/local/$CHECKPOINT
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/local/$CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_sample_100.json --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type 'cpu' --wandb_status online --image_size 384

    # Exp 2. WCE 
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384
    # conda activate cqa-wce
    # sbatch models-hf/ada-script-blip-wce-384.sh


    # Exp3. OCR post + desc + 384
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online  --image_size 384 --ocr 1 --desc 1
    # conda activate cqa-size
    # sbatch models-hf/ada-script-blip-post-384.sh


    # Exp 4. OCR post + desc + wce - all
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --ocr 1 --desc 1
    # conda activate cqa-wce 
    # sbatch models-hf/ada-script-blip-all-384.sh


    # Exp 5. BBOX 
    # python models-hf-bbox/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --bbox 1
    # conda activate cqa-size
    # sbatch models-hf/ada-script-blip-bbox-384.sh


    # Exp 6. BBOX segment
    # python models-hf-bbox/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox_segment.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --bbox_segment 1
    # conda activate cqa-size
    # sbatch models-hf/ada-script-blip-bbox-seg-384.sh

    # Exp 7. OCR 384
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE  \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 0 # # for image size experiment 
    # conda activate cqa-size
    # sbatch models-hf/ada-script-blip-ocr-384.sh

    # Exp 8 . DESC 384
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 0 --desc 1
    # conda activte cqa-size
    # sbatch models-hf/ada-script-blip-desc-384.sh

    # Exp 9. 384 + ocr + desc (pre)
    # python models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE \
    #     --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_adv.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE  \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1
    # conda activte cqa-size
    # sbatch models-hf/ada-script-blip-pre-384.sh

    # Exp 10. BLIP - 384 + ocr + desc (post) + bbox_segment (all-bboxsegments) + wce
    # python models-hf-bbox/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4 --train_dir datasets/$DATASET_SIZE  \
    #     --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    #     --question_dir datasets/questions/all/master_bbox_segment_adv_ocr.json  --experiment_name ada-$MODEL-$EXP_NAME-$DATE \
    #     --ngpus 1 --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --ocr 1 --desc 1 --bbox_segment 1
    # conda activate cqa-wce 
    # sbatch models-hf/ada-script-blip-all-bboxseg-384.sh


elif [ "$RUN_TYPE" = "inference" ]; then 
    echo "Running Inference"
    #rm -r datasets/results/$MODEL/$EXP_NAME

    # Exp 1 . base 384
    export NUM_NODES=1
    # export EPOCHS=10
    export LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES=0
    
    #torchrun --nnodes 1 --nproc_per_node $NUM_GPUS --rdzv_id=31459 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29900 \
    
    #accelerate launch --multi_gpu --num_processes=$NUM_GPUS 
    
    # accelerate launch --num_processes=$NUM_GPUS --main_process_port 29501 python models-hf/blip-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
    #         --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME  --machine_type $MACHINE_TYPE  --ocr 1 # --image_size 384
    # conda activate cqa-base
    # sbatch models-hf-bbox/infer-blip-base-384.sh

    # Exp 2 . wce 384
    # python models-hf/blip-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master.json \
    #         --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE --wce 1 
    # conda activate cqa-wce
    # sbatch models-hf-bbox/infer-blip-wce-384.sh


    # Exp 3 ocr pre
    accelerate launch  --num_processes=$NUM_GPUS  python models-hf-bbox/blip-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
            --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type $MACHINE_TYPE --ocr 1 
    # conda activate cqa-base
    # sbatch models-hf-bbox/ada-script-pre.sh

    # Exp 3 ocr post
    # accelerate launch  --num_processes=$NUM_GPUS  python models-hf-bbox/blip-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
    #         --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type $MACHINE_TYPE --ocr 1 
    # conda activate cqa-base
    # sbatch models-hf-bbox/ada-script-post.sh


    # Exp 4 . all
    # python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv_ocr.json \
    #         --results_dir datasets/results/$MODEL_$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 384  --ocr 1 --desc 1
    # conda activate cqa-wce
    # sbatch models-hf/infer-blip-all-384.sh

    # Exp 5. Bbox
    # python models-hf-bbox/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_bbox.json  \
    #         --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE --image_size 384 --bbox 1
    # conda activate cqa-size
    # sbatch models-hf-bbox/ada-script.sh

    # Exp 6 BBOX seg
    # python models-hf-bbox/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_bbox_segment.json  \
    #         --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE --image_size 384 --bbox_segment 1
    # conda activate cqa-size
    # sbatch models-hf-bbox/ada-script-384-bbox-seg.sh

    # Exp 7 OCR 
    # python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
    #       --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 384  --ocr 1 --desc 0
    # conda activate cqa-size
    # sbatch models-hf/infer-blip-ocr-384.sh
    
    # Exp 8 Desc
    # python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
    #        --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 384  --ocr 0 --desc 1
    # conda activate cqa-size
    # sbatch models-hf/infer-blip-desc-384.sh
    
    # Exp 9 OCR + Desc (Pre)
    # python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_adv.json \
    #        --results_dir datasets/results/$MODEL/$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 384  --ocr 1 --desc 1
    #  conda activate cqa-size
    # sbatch models-hf/infer-blip-pre-384.sh

    # Exp 10. BLIP - 384 + ocr + desc (post) + bbox_segment (all-bboxsegments) + wce
    # python models-hf/$MODEL-vqa-inference.py --test_dir datasets/$DATASET_SIZE --test_batch_size 4  --checkpoint_dir  $CHECKPOINT --question_dir datasets/questions/all/master_bbox_segment_adv_ocr.json \
    #        --results_dir datasets/results/$MODEL/$MACHINE_TYPE/$DATASET_SIZE/$EXP_NAME --machine_type $MACHINE_TYPE  --image_size 384  --ocr 1 --desc 1 --bbox_segment 1
    # # conda activate cqa-wce
    # sbatch models-hf/infer-blip-all-bboxseg-384.sh

else
    echo "Not valid"
fi 

