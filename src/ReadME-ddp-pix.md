## pix EXPERIMENTS

SIZE='384a'
export CHECKPOINT_DIR_PIX=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/pix/$SIZE-ddp
export RESULTS_DIR_PIX=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-pix-hf/results-ddp/$SIZE
export LOCAL_SCRIPTS=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf
export ADA='rahul.mehta@ada:circuitQA'


scp -r $LOCAL_SCRIPTS/pix-files/original $ADA/models-hf/pix-files
ls models-hf/pix-files/original
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-base.sh $ADA/models-hf
scp $LOCAL_SCRIPTS/pix-vqa-train-ddp.py $ADA/models-hf/
scp -r $LOCAL_SCRIPTS/pix-files/wce $ADA/models-hf/pix-files

scp -r $LOCAL_SCRIPTS/pix-files/configuration_pix2struct.py $ADA/models-hf/pix-files



scp -r datasets/questions/all/master_adv_ocr.json $ADA/datasets/questions/all
scp -r datasets/questions/all/master_bbox_segment.json $ADA/datasets/questions/all
#### Base 
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-base.sh $ADA/models-hf
conda activate cqa-wce
sbatch models-hf/pix-base.sh
cat runs/pix/pix-base-384.txt

scp -r $ADA/checkpoints-pix-ddp-base-384-10Jan $CHECKPOINT_DIR_PIX

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-base.sh $ADA/models-hf/pix-base-infer.sh
sbatch models-hf/pix-base-infer.sh
cat runs/pix/pix-base-infer.txt


scp -r $ADA/datasets/results/pix/ddp/384a/base $RESULTS_DIR_pix         

#### Base + LR 
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-base-lr.sh $ADA/models-hf
scp $LOCAL_SCRIPTS/pix-vqa-train-ddp.py $ADA/models-hf/


conda activate cqa-wce
sbatch models-hf/pix-base-lr.sh

cat runs/pix/pix-base-lr-384.txt

scp -r $ADA/checkpoints-pix-ddp-base-lr-384-4Jan $CHECKPOINT_DIR_PIX
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-base-lr.sh $ADA/models-hf/pix-base-lr-infer.sh
sbatch models-hf/pix-base-lr-infer.sh

scp -r $ADA/datasets/results/pix/ddp/384a/base-lr $RESULTS_DIR_PIX     


#### WCE 
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-wce.sh $ADA/models-hf/pix-wce.sh
scp $LOCAL_SCRIPTS/pix-vqa-train-ddp.py $ADA/models-hf/
scp $LOCAL_SCRIPTS/pix-files/pix/wce $ADA/models-hf/pix-files/pix
scp $LOCAL_SCRIPTS/pix-files/original $ADA/models-hf/pix-files/original

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-wce.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/pix-wce.sh
cat runs/pix/pix-wce.txt

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-wce.sh $ADA/models-hf/pix-wce-infer.sh
scp $LOCAL_SCRIPTS/pix-vqa-inference.py $ADA/models-hf/

conda activate cqa-wce
sbatch models-hf/pix-wce-infer.sh
cat runs/pix/pix-wce-infer.txt

scp -r $ADA/datasets/results/pix/ddp/384a/wce $RESULTS_DIR_PIX     
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/00-evaluate-pred.py

# predict
scp -r $ADA/checkpoints-pix-ddp-wce-4Jan $$CHECKPOINT_DIR_PIX
scp -r $ADA/datasets/results/pix/ddp/384a/wce $RESULTS_DIR_PIX     


#### OCR pre
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-ocr-pre.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/pix-ocr-pre.sh
cat runs/pix/pix-ocr-pre.txt

scp $CHECKPOINT_DIR_PIX/checkpoints-pix-ddp-ocr-pre-384-24Jan.zip $ADA
scp $CHECKPOINT_DIR_PIX/checkpoints-pix-ddp-desc-384-24Jan.zip $ADA
scp $CHECKPOINT_DIR_PIX/checkpoints-pix-ddp-bbox-384-24Jan.zip $ADA

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-ocr-pre.sh $ADA/models-hf/pix-ocr-pre-infer.sh
sbatch models-hf/pix-ocr-pre-infer.sh
cat runs/pix/pix-ocr-pre-384-infer.txt

scp -r $ADA/datasets/results/pix/ddp/384a/ocr-pre $RESULTS_DIR_PIX   

### OCR post
scp $LOCAL_SCRIPTS/pix-vqa-train-ddp.py $ADA/models-hf

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-ocr-post.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/pix-ocr-post.sh
cat runs/pix/pix-ocr-post-384.txt

scp -r $CHECKPOINT_DIR_PIX/checkpoints-pix-ddp-ocr-post-384-31Jan $ADA

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-ocr-post.sh $ADA/models-hf/pix-ocr-post-infer.sh

sbatch models-hf/pix-ocr-post-infer.sh
cat runs/pix/pix-ocr-post-infer.txt

scp -r $ADA/datasets/results/pix/ddp/384a/ocr-post $RESULTS_DIR_PIX                 
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py


### DESC
scp $LOCAL_SCRIPTS/pix-vqa-inference.py $ADA/models-hf
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-desc.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/pix-desc.sh
cat runs/pix/pix-desc-384.txt


scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-desc.sh $ADA/models-hf/pix-desc-infer.sh
sbatch models-hf/pix-desc-infer.sh
cat runs/pix/pix-desc-384-infer.txt

scp -r $ADA/datasets/results/pix/ddp/384a/desc $RESULTS_DIR_PIX   


## BBOX 

scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-bbox.sh $ADA/models-hf/pix-bbox-infer.sh
sbatch models-hf/pix-bbox-infer.sh
cat runs/pix/pix-bbox-384-infer.txt

scp -r $ADA/datasets/results/pix/ddp/384a/bbox $RESULTS_DIR_PIX   


### BBOX SEGMENT
scp $LOCAL_SCRIPTS/ada-scripts/pix/pix-bbox-segment.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/pix-bbox-segment.sh
cat runs/pix/pix-bbox-segment-384.txt


scp -r $CHECKPOINT_DIR_PIX/checkpoints-pix-ddp-bbox-segment-384-31Jan $ADA
scp -r $ADA/datasets/results/pix/ddp/384a/bbox-segment $RESULTS_DIR_PIX   
