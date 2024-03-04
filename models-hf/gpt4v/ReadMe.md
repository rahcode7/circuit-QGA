
#### Input data prep for the model


```
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv.json --op_path models-hf/gpt4v/datasets/ocr --exp_name ocr
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_adv_ocr.json --op_path models-hf/gpt4v/datasets/ocr-post --exp_name ocr-post
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox.json --op_path models-hf/gpt4v/datasets/bbox --exp_name bbox
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment.json --op_path models-hf/gpt4v/datasets/bbox_segment --exp_name bbox_segment
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_yolo.json --op_path models-hf/gpt4v/datasets/bbox_yolo --exp_name bbox_yolo
python models-hf/gpt4v/data-prep.py --q_path datasets/questions/all/master_bbox_segment_yolo.json --op_path models-hf/gpt4v/datasets/bbox_segment_yolo --exp_name bbox_segment_yolo
```


#### Post processing of the model outputs

##### Prepare predictions.json file by merging multiple outputs file
```
python post-process-0.py
```


##### Prepare predictions-final.json file by merging multiple outputs file
```
python models-hf/gpt4v/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-gpt4v-hf/results-ddp/384a/desc  --exp_name desc
```

#### Evaluations - Accuracy calculate
```
python src/evaluate/00-evaluate-pred.py 
```

#### Evaluations - HS calculate
```
python src/evaluate/02-a-hallucination.py 
```