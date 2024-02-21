## InstructBLIP

### Step 1 Run Experiments 

```
pip install bitsandbytes
cd circuitQA
git pull
conda activate <circuitQAenvironment>
mkdir datasets/results/InstructBLIP
```

#### BASE model 
```
MODEL='InstructBLIP'
EXP_NAME='base'
DATASET_SIZE='384a'

export CUDA_VISIBLE_DEVICES=0
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

#### DESC model 
```
MODEL='InstructBLIP'
EXP_NAME='desc'
DATASET_SIZE='384a'
export CUDA_VISIBLE_DEVICES=1
python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

#### OCR PRE model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-pre'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

#### OCR POST model 
```
MODEL='InstructBLIP'
EXP_NAME='ocr-post'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_adv_ocr.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

#### BBOX-YOLO model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME
```

#### BBOX-Segment-YOLO  model 
```
MODEL='InstructBLIP'
EXP_NAME='bbox-segment'
DATASET_SIZE='384a'

python models-hf/models-InstructBLIP-hf/iblip-eval-single.py --question_dir datasets/questions/all/master_bbox_segment_yolo.json  \
    --image_dir datasets/$DATASET_SIZE --results_dir datasets/results/InstructBLIP/$EXP_NAME --exp_name $EXP_NAME

```

### Step 2 Upload Experiments here
