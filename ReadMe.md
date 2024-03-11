
## Section 1: Dataset Preparation Guide


### 1. Prepare master dataset of images and metadata
##### Step 1 Unify all 5 datasets
python src/data-prep/02-data-prep-master.py 


##### Step 2 Identify and remove duplicate images 
python src/data-prep/03-duplicate-identify.py 
python src/data-prep/03-duplicate-remove.py

##### Step 3 Split datasets
python src/data-prep/04-split-dataset.py

##### Step 4 Map classes
python src/data-prep/05-class-mapping.py

### 2. Prepare Questions-Answers for various question types
##### Prepare count based questions
python src/question-generation/count-based/Q-count.py

##### Prepare spatial count based questions
python src/question-generation/count-based/Q-count-complex.py

##### Prepare junction based questions
python src/question-generation/junction-based/Q-junction.py

##### Prepare position based questions
python src/question-generation/junction-based/Q-position.py

##### Prepare value based questions
python src/question-generation/value-based/00-bounding-box.py
python src/question-generation/value-based/01-dist-calc.py
python src/question-generation/value-based/02-Q-value-based.py

### 3. Prepare master VQA datasets
##### Prepare master VQA dataset
python src/question-generation/master-data.py

##### Prepare master VQA dataset for OCR and Description experiments
python src/question-generation/master-data-desc-ocr.py

##### Prepare master VQA dataset for Bounindg box experiments
python src/question-generation/master-data-bbox.py

##### Prepare master VQA dataset for Bounindg box segments experiments
python src/question-generation/master-data-bbox-segment.py

##### Prepare class weights for weighted cross entropy experiments
python src/question-generation/master-data-classweights.py


## Section 2 : Run Generative - Fine tuning and instruction tuned models 


#### Install conda environment
```
conda create --name cqa-size python=3.8.17  
conda activate cqa-size
conda install pip  
pip install -r requirements.txt   

```

###### Transformers versions
For BLIP and GIT - 4.30.2
For Pix2Struct - 4.36.0


#### Image Original sizes

```
gdown https://drive.google.com/uc\?id\=1523cKJ6sfZN9k4eSYBdx85gHpR41qNbo   
gdown https://drive.google.com/uc\?id\=17PqgVlh-a63XTOS55PPEsmfQyXGEHPFu

tar -xvf model-inputs.tar  -C  datasets
tar -xvf model-inputs-jn.tar  -C  datasets
rm model-inputs.tar model-inputs-jn.tar 
```

## Download images and questions
#### Images Resized (384a)
```
gdown https://drive.google.com/uc\?id\=1WXo-LLTmVO6iDAJK45TYsyVm07JIgTF-
tar -xvf 384a.tar -C datasets
rm 576a.tar
```

#### Download Question datasets
```
gdown https://drive.google.com/uc\?id\=1dDbX41Ty-efQk4bm5EADd4HF6Or9t_Ig
mv master.json datasets/questions/all

gdown https://drive.google.com/uc\?id\=1X55cNJHKMD6YsmpjqknniBRF2UwJwJvK
mv master_adv.json datasets/questions/all

gdown https://drive.google.com/uc\?id\=12erEPkuSdUVUU-4SDqA9draCZdYxDOpo
mv master_adv_ocr.json datasets/questions/all

gdown https://drive.google.com/uc\?id\=1SH3fgBqymrF66YTGv4fAj4bm7yEtd0yi
mv master_bbox.json datasets/questions/all

gdown https://drive.google.com/uc\?id\=1Anc1-jyzH0gpzECvq4Ad2nL8s9oUKbnx
mv master_bbox_segment.json datasets/questions/all

gdown https://drive.google.com/uc\?id\=15mUNd15FokT9Y81dAOHW2qp_h7kbxZhD

```



### Distributed set up 
```
pip install bitsandbytes scipy accelerate
```
### Fine Tuned generative models 

#### PIX (Distributed) - 384

#### Experiment 1 PIX LR 
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base-lr' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='19Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --lr 1 --wce 0 
```

#### Experiment 2 PIX WCE 

```

MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='wce' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='19Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1
```

#### Experiment 3 PIX OCR PRE
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-pre' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --ocr 1 
```


#### Experiment 4 PIX OCR POST
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-post' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='28Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --ocr 1
```

#### Experiment 5 PIX DESC
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='desc' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_adv_ocr.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --desc 1
```

#### Experiment 6 PIX BBOX
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_bbox.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --bbox 1
```

#### Experiment 7 PIX BBOX SEGMENT
```
MODEL='pix'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox-segment' # wce,size,focal,desc,ocr,ocr-post,ocr-desc,size-768,post-576,base-384,post-384,base-384
RUN_TYPE='train' # train,inference
DATE='24Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
echo $CHECKPOINT
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 16 --val_batch_size 16 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_bbox_segment.json  --experiment_name ms-$MODEL-$EXP_NAME \
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --wce 1 --bbox_segment 1

```

### Instruction Fine Tuned models

#### 1. GPT4 Experiments

#### Step 1 Input data prep for the model


```
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv.json --op_path models-hf/gpt4v/datasets/ocr --exp_name ocr --hosted_url "https://rahcode7.github.io/"
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv_ocr.json --op_path models-hf/gpt4v/datasets/ocr-post --exp_name ocr-post
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox.json --op_path models-hf/gpt4v/datasets/bbox --exp_name bbox
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment.json --op_path models-hf/gpt4v/datasets/bbox_segment --exp_name bbox_segment
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_yolo.json --op_path models-hf/gpt4v/datasets/bbox_yolo --exp_name bbox_yolo
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment_yolo.json --op_path models-hf/gpt4v/datasets/bbox_segment_yolo --exp_name bbox_segment_yolo
```


#### Step 2  Post processing of the model outputs

##### Prepare predictions.json file by merging multiple outputs file
```
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name ocr
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name ocr-post
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_segment
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_yolo
python models-hf/gpt4v/post-process-0.py --prediction_dir models-gpt4v-hf/results-ddp/384a  --exp_name bbox_segment_yolo


```


##### Step 3 repare predictions-final.json file by merging multiple outputs file
```
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/ocr  --exp_name ocr
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/ocr-post  --exp_name ocr-post
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox  --exp_name bbox
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_segment  --exp_name bbox_segment
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_yolo  --exp_name bbox_yolo
python models-hf/gpt4v/post-process.py --prediction_dir models-gpt4v-hf/results-ddp/384a/bbox_segment_yolo  --exp_name bbox_segment_yolo


```

#### Step 4.1  Evaluations - Accuracy calculate
```
python src/evaluate/00-evaluate-pred.py 
```

#### Step 4.2 Evaluations - HS calculate
```
python src/evaluate/02-a-hallucination.py 
```


<!-- #### 2. LLaVA Fine tuning

#### Step 0 (If you don't have LLaVA)
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e . -->



<!-- #### Step 1 (fine tuning codes and model weights)
```
pip install deepspeed
pip install flash-attn==2.3.3
pip install flash-attn --no-build-isolation

# pip install torch-2.0.1



git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b  # get weights
git clone https://github.com/bdytx5/finetune_LLaVA.git # get code 
cp LLaVA/scripts/zero2.json finetune_LLaVA/scripts/zero2.json 

```


#### Step 2 : Get instruction train/val datasets


```
pip install gdown==v4.6.0   # upgrade to latest version
mkdir datasets/llava
gdown https://drive.google.com/drive/folders/1GMrFlCBH7utOaP8zKsqZFchabisu0qdE -O datasets/llava --folder

```

#### Step 3 : Script for finetuning
```
MODEL='llava'
EXP_NAME='base'
DATE='5Mar'
CHECKPOINT="checkpoints-$MODEL-$EXP_NAME-$DATE"
mkdir $CHECKPOINT
export CUDA_VISIBLE_DEVICES=0

deepspeed LLaVA/llava/train/train_mem.py \
        --deepspeed LLaVA/scripts/zero2.json \
        --lora_enable True \
        --lora_r 128 \
        --lora_alpha 256 \
        --mm_projector_lr 2e-5 \
        --bits 4 \
        --model_name_or_path LLaVA/llava/llava-v1.5-7b \
        --version llava_llama_2 \
        --data_path datasets/llava/train/dataset.json \
        --validation_data_path datasets/llava/val/dataset.json \
        --image_folder datasets/384a/ \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 False \
        --output_dir $CHECKPOINT \
        --num_train_epochs 5 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy epoch --save_strategy steps \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 False \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb  
``` -->


#### InstructBLIP - ZERO SHOT

##### Step 1 Run Experiments 

```
pip install bitsandbytes
cd circuitQA
git pull
conda activate <circuitQAenvironment>
mkdir datasets/results/InstructBLIP
```

##### BASE model 
```
MODEL='InstructBLIP'
EXP_NAME='base'
DATASET_SIZE='384a'

# MS
export CUDA_VISIBLE_DEVICES=0
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

# Ada
python models-hf/models-InstructBLIP-hf/iblip-eval-single-mac.py --question_dir ../datasets/questions/all/master_adv_ocr.json  \
    --image_dir ../datasets/$DATASET_SIZE --results_dir ../datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

##### DESC model 
```
MODEL='InstructBLIP'
EXP_NAME='desc'
DATASET_SIZE='384a'
export CUDA_VISIBLE_DEVICES=1
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### OCR PRE model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-pre'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### OCR POST model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-post'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
``` 

##### BBOX-YOLO model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-yolo'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### BBOX-Segment-YOLO  model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-segment-yolo'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_segment_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

##### BBOX-ORACLE model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

##### BBOX-Segment-ORACLE model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-segment'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_segment.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```


###### Step 2 Post process
python circuit-QGA/models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir models-InstructBLIP-hf/results-ddp/384a --exp_name base
python circuit-QGA/models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir models-InstructBLIP-hf/results-ddp/384a --exp_name desc



