## GIT EXPERIMENTS

SIZE='384a'
export CHECKPOINT_DIR_BLIP=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/$SIZE-ddp
export CHECKPOINT_DIR_GIT=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/GIT/$SIZE-ddp
export RESULTS_DIR_BLIP=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results-ddp/$SIZE
export RESULTS_DIR_GIT=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-GIT-hf/results-ddp/$SIZE
export LOCAL_SCRIPTS=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf
export ADA='rahul.mehta@ada:circuitQA'


scp -r $LOCAL_SCRIPTS/git-files/original $ADA/models-hf/git-files
ls models-hf/git-files/original
scp $LOCAL_SCRIPTS/ada-scripts/git/git-base.sh $ADA/models-hf
scp $LOCAL_SCRIPTS/git-vqa-train-ddp.py $ADA/models-hf/
scp -r $LOCAL_SCRIPTS/git-files/wce $ADA/models-hf/git-files

#### Base 
scp $LOCAL_SCRIPTS/ada-scripts/git/git-base.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-base.sh
cat runs/git/git-base.txt

scp -r $ADA/checkpoints-git-ddp-base-30Dec $CHECKPOINT_DIR_GIT

scp $LOCAL_SCRIPTS/ada-scripts/git/git-base.sh $ADA/models-hf/git-base-infer.sh
sbatch models-hf/git-base-infer.sh
cat runs/git/git-base-infer.txt

scp -r $ADA/datasets/results/git/ddp/384a/base $RESULTS_DIR_GIT         

#### Base + LR 
scp $LOCAL_SCRIPTS/ada-scripts/git/git-base-lr.sh $ADA/models-hf
scp $LOCAL_SCRIPTS/git-vqa-train-ddp.py $ADA/models-hf/

conda activate cqa-base
sbatch models-hf/git-base-lr.sh

cat runs/git/git-base-lr.txt

scp -r $ADA/checkpoints-git-ddp-base-lr-384-4Jan $CHECKPOINT_DIR_GIT
scp $LOCAL_SCRIPTS/ada-scripts/git/git-base-lr.sh $ADA/models-hf/git-base-lr-infer.sh
sbatch models-hf/git-base-lr-infer.sh

scp -r $ADA/datasets/results/git/ddp/384a/base-lr $RESULTS_DIR_GIT         


#### WCE 
scp $LOCAL_SCRIPTS/ada-scripts/git/git-wce.sh $ADA/models-hf/git-wce.sh
scp $LOCAL_SCRIPTS/git-vqa-train-ddp.py $ADA/models-hf/
scp $LOCAL_SCRIPTS/git-files/git/wce $ADA/models-hf/git-files/git
scp $LOCAL_SCRIPTS/git-files/original $ADA/models-hf/git-files/original

scp $LOCAL_SCRIPTS/ada-scripts/git/git-wce.sh $ADA/models-hf
conda activate cqa-wce
sbatch models-hf/git-wce.sh
cat runs/git/git-wce.txt

scp $LOCAL_SCRIPTS/ada-scripts/git/git-wce.sh $ADA/models-hf/git-wce-infer.sh
scp $LOCAL_SCRIPTS/git-vqa-inference.py $ADA/models-hf/

conda activate cqa-wce
sbatch models-hf/git-wce-infer.sh
cat runs/git/git-wce-infer.txt

# predict
scp -r $ADA/checkpoints-git-ddp-wce-4Jan $CHECKPOINT_DIR_GIT
scp -r $ADA/datasets/results/git/ddp/384a/wce $RESULTS_DIR_GIT         


#### OCR pre
scp -r $CHECKPOINT_DIR_GIT/checkpoints-git-ddp-ocr-pre-384-9Jan $ADA

scp $LOCAL_SCRIPTS/ada-scripts/git/git-ocr-pre.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-ocr-pre.sh
cat runs/git/git-ocr-pre-infer.txt

scp -r $ADA/datasets/results/git/ddp/384a/ocr-pre $RESULTS_DIR_GIT 

### OCR post
scp $LOCAL_SCRIPTS/ada-scripts/git/git-ocr-post.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-ocr-post.sh
cat runs/git/git-ocr-post.txt

scp -r $ADA/checkpoints-git-ddp-ocr-post-384-6Jan $CHECKPOINT_DIR_GIT
scp $LOCAL_SCRIPTS/ada-scripts/git/git-ocr-post.sh $ADA/models-hf/git-ocr-post-infer.sh

sbatch models-hf/git-ocr-post-infer.sh
cat runs/git/git-ocr-post-infer.txt

scp -r $ADA/datasets/results/git/ddp/384a/ocr-post $RESULTS_DIR_GIT                 
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py

### DESC
scp -r $CHECKPOINT_DIR_GIT/checkpoints-git-ddp-desc-384-10Jan $ADA
scp $LOCAL_SCRIPTS/ada-scripts/git/git-desc.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-desc.sh
cat runs/git/git-desc-384.txt



scp -r $ADA/datasets/results/git/ddp/384a/desc $RESULTS_DIR_GIT 

### BBOX SEGMENT
scp -r $CHECKPOINT_DIR_GIT/checkpoints-git-ddp-bbox-segment-384-7Jan $ADA
scp $LOCAL_SCRIPTS/ada-scripts/git/git-bbox-segment.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-bbox-segment.sh
cat runs/git/git-bbox-segment-384.txt

scp -r $ADA/datasets/results/git/ddp/384a/bbox-segment $RESULTS_DIR_GIT 


### BBOX
scp -r $CHECKPOINT_DIR_GIT/checkpoints-git-ddp-bbox-384-7Jan $ADA
scp $LOCAL_SCRIPTS/ada-scripts/git/git-bbox.sh $ADA/models-hf
conda activate cqa-base
sbatch models-hf/git-bbox.sh
cat runs/git/git-bbox-384.txt


cat runs/git/git-bbox-384-infer.txt


### All
scp -r $ADA/datasets/results/git/ddp/384a/all $RESULTS_DIR_GIT 
scp $LOCAL_SCRIPTS/ada-scripts/git/git-all.sh $ADA/models-hf
sbatch models-hf/git-all.sh
cat runs/blip/git-all-384-infer.txt

python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/00-evaluate-pred.py

