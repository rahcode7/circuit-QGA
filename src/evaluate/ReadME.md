### Calculate Hallucination

#### 1.1For BLIP,GIT,PIX

```
MODEL='BLIP'
PREDICTION_FILE="predictions.csv"
exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all','bbox-yolo','bbox-segment-yolo'] # BLIP
```

#### 1.2 For LLAVA
```
conda activate llava
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/cqa-llava/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-LLaVa-hf/results-ddp/384a/bbox --exp_name ocr-pre
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/00-evaluate-pred.py

MODEL="LLaVA"
PREDICTION_FILE="predictions-final.csv"
exp_list = ['ocr-post','base','desc','bbox']
```

### 2. Run
```
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/02-hallucination.py
```


