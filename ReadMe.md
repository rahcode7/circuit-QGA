### BLIP Model 

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

####  Wandb
wandb login 
`ce18e8ae96d72cd78a7a54de441e9657bc0a913d`


#### Image Original sizes

```
gdown https://drive.google.com/uc\?id\=1523cKJ6sfZN9k4eSYBdx85gHpR41qNbo   
gdown https://drive.google.com/uc\?id\=17PqgVlh-a63XTOS55PPEsmfQyXGEHPFu

tar -xvf model-inputs.tar  -C  datasets
tar -xvf model-inputs-jn.tar  -C  datasets
rm model-inputs.tar model-inputs-jn.tar 
```

## Download images and questions

#### Images Resized (576)
```
gdown https://drive.google.com/uc\?id\=10nq-bjxAKt_szWEe8FeaM4k3lVA3rie3
tar -xvf 576.tar -C datasets
rm 576.tar
```

#### Images Resized (576a)
```
gdown https://drive.google.com/uc\?id\=19_12tIf0kKDdRO43XPj7BgyqmEwZp5NG
tar -xvf 576a.tar -C datasets
rm 576a.tar
```

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
<!-- #### BLIP final set of experiments (Can be run in a single conda environment)
- Create SITE_PACKAGES_PATH if needed

For MS
```
export SITE_PACKAGES_PATH='/scratch/AzureNfsServer_INPUT1/vc_data/users/gmanish/condaEnv/circuitvqa/lib/python3.9/site-packages'
```

For ADA
```
export SITE_PACKAGES_PATH='/home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages'
```


First we need to run this experiment -->

### Distributed set up 
```
pip install bitsandbytes scipy accelerate
```


##  BLIP (Distributed) - 576 
##### Experiment 1. base + lr 

```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base-lr' # 
RUN_TYPE='train' # train,inference
DATE='6Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 0 --lr 1

```

##### Experiment 2. wce + ocr-pre

```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-pre' # 
RUN_TYPE='train' # train,inference
DATE='8Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
--question_dir datasets/questions/all/master_adv.json  --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 1 --ocr 1

```

##### Experiment 3. wce + ocr-post

```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-post' # 
RUN_TYPE='train' # train,inference
DATE='8Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
--question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 1 --ocr 1

```

##### Experiment 4 wce + desc 

```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='desc' # 
RUN_TYPE='train' # train,inference
DATE='8Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
--question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 1 --desc 1

```

##### Experiment 5. wce + bbox

```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox' # 
RUN_TYPE='train' # train,inference
DATE='8Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  \
--question_dir datasets/questions/all/master_bbox.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 1 --bbox 1

```

##### Exp 6. bbox-segment
```
MODEL='blip'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox-segment' # 
RUN_TYPE='train' # train,inference
DATE='8Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Set your gpu ids 

accelerate launch --num_processes=$NUM_GPUS models-hf/blip-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 8 --val_batch_size 8 --train_dir datasets/$DATASET_SIZE \
    --val_dir datasets/$DATASET_SIZE --checkpoint_dir  $CHECKPOINT  \
    --question_dir datasets/questions/all/master_bbox_segment.json --experiment_name ms-$MODEL-$EXP_NAME-$DATE \
    --ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --accumulation_steps 4 --image_size 576 --wce 1 --bbox_segment 1

```

## GIT (Distributed) - 576

### Experiment 1 GIT LR 
```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base-lr' # 
RUN_TYPE='train' # train,inference
DATE='16Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 0 --lr 1
```


### Experiment 2 GIT WCE
```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='wce' # 
RUN_TYPE='train' # train,inference
DATE='18Jan'
SIZE='576'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='576a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 576 --accumulation_steps 4 --wce 0 --lr 1
```



## GIT (Distributed) - 384

#### Experiment 1 GIT Base 

```
MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base' # 
RUN_TYPE='train' # train,inference
DATE='4Jan'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=3

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2 # Set your gpu ids 

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 0 --lr 0
```

#### Experiment 2 GIT Base + LR 

```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='base-lr' # 
RUN_TYPE='train' # train,inference
DATE='4Jan'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=3

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 0 --lr 1
```

#### Experiment 3 GIT WCE
```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='wce' # 
RUN_TYPE='train' # train,inference
DATE='5Jan'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=3

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2


accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --wce 1 --lr 0
```

#### Experiment 4 GIT LR + OCR Pre

```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr' # 
RUN_TYPE='train' # train,inference
DATE='9Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master_adv.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --lr 1 --ocr 1

```


<!-- #### Experiment 5 GIT Base + OCR Post

```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='ocr-post' # 
RUN_TYPE='train' # train,inference
DATE='5Jan'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=3

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 4 --val_batch_size 4  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --ocr 1
```
 -->

#### Experiment 6 GIT LR + Desc

```
MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='desc' # 
RUN_TYPE='train' # train,inference
DATE='10Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master_adv_ocr.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --lr 1 --desc 1

```

#### Experiment 7 GIT LR + BBox

```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox' # 
RUN_TYPE='train' # train,inference
DATE='7Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master_bbox.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --lr 1 --bbox 1

```

#### Experiment 8 GIT LR + BBox Segment

```

MODEL='git'  # git,blip
MACHINE_TYPE='ddp' # ddp or dp or cpu
EXP_NAME='bbox-segment' # 
RUN_TYPE='train' # train,inference
DATE='7Jan'
SIZE='384'
CHECKPOINT="checkpoints-$MODEL-$MACHINE_TYPE-$EXP_NAME-$SIZE-$DATE"
DATASET_SIZE='384a'
NUM_GPUS=4

rm -rf $CHECKPOINT
mkdir $CHECKPOINT

export NUM_NODES=1
export EPOCHS=10
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set your gpu ids

accelerate launch --multi_gpu --num_processes=$NUM_GPUS models-hf/$MODEL-vqa-train-$MACHINE_TYPE.py --num_epochs $EPOCHS --train_batch_size 10 --val_batch_size 10  --train_dir datasets/$DATASET_SIZE \
--val_dir datasets/$DATASET_SIZE  --checkpoint_dir  $CHECKPOINT  --learning_rate 1e-5 \
--question_dir datasets/questions/all/master_bbox_segment.json   --experiment_name ms-$MODEL-$EXP_NAME  \
--ngpus $NUM_GPUS --machine_type $MACHINE_TYPE --wandb_status online --image_size 384 --accumulation_steps 4 --lr 1 --bbox_segment 1
```


# PIX (Distributed) - 384

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
    --ngpus 4 --machine_type $MACHINE_TYPE --wandb_status online --max_patches 512 --accumulation_steps 4 --lr 1
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