 # BLIP

# 
vi $SITE_PACKAGES_PATH/transformers/models/blip/modeling_blip.py 
c

## E1 base
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/
conda activate cqa-size
sbatch models-hf/blip-base-384.sh


## E3 wce 
## WCE SET Up
SITE_PACKAGES_PATH='/home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages'
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/blip-files/modeling_blip_text.py rahul.mehta@ada:circuitQA/models-hf/blip-files
conda activate cqa-wce
cp models-hf/blip-files/modeling_blip_text.py $SITE_PACKAGES_PATH/transformers/models/blip/

#### Train 
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-wce-384.sh

conda activate cqa-wce
sbatch models-hf/blip-wce-test-384.sh

# Inferences  
# Exp 4 . all

conda activate cqa-wce
sbatch models-hf/infer-blip-all-384.sh

# Exp 7 OCR 
conda activate cqa-size
sbatch models-hf/infer-blip-ocr-384.sh

# Exp 8 DESC 
conda activate cqa-size
sbatch models-hf/infer-blip-desc-384.sh

# Training  
# Exp 9 
conda activte cqa-size
sbatch models-hf/ada-script-blip-pre-384.sh
sbatch models-hf/infer-blip-pre-384.sh

scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/pre-384 /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results/384

############### Exp 10
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/all/master_bbox_segment_adv_ocr.json rahul.mehta@ada:circuitQA/datasets/questions/all/
scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/ada-script-blip-all-bboxseg-384.sh

conda activate cqa-wce 
sbatch models-hf/ada-script-blip-all-bboxseg-384.sh

## inference
conda activate cqa-wce 
scp models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/infer-blip-all-bboxseg-384.sh
scp -r rahul.mehta@ada:circuitQA/checkpoints-dp-all-bboxseg-384-3Dec/ /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/384

scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/all-bboxseg-384/


GIT
# Inferences

####################  E 6 bbox-seg-384

# 1. Transfer
scp -r rahul.mehta@ada:circuitQA/checkpoints-dp-git-bbox-seg-384-1Dec/  /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/GIT/384
scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-git.sh rahul.mehta@ada:circuitQA/models-hf/infer-git-bbox-seg-384.sh

# 2. run inference
conda activate cqa-git
sbatch models-hf/infer-git-bbox-seg-384.sh 

# 3. collect results
scp -r rahul.mehta@ada:circuitQA/datasets/results/git/bbox-seg-384 /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-GIT-hf/results/384


##### E5  GIT - bbox-384 INFERENCE

# 1. Transfer
scp -r rahul.mehta@ada:circuitQA/checkpoints-dp-git-bbox-384-1Dec/  /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/GIT/384
scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-git.sh rahul.mehta@ada:circuitQA/models-hf/infer-git-bbox-384.sh

# 2. run inference
conda activate cqa-git
sbatch models-hf/infer-git-bbox-384.sh 

# 3. collect results
scp -r rahul.mehta@ada:circuitQA/datasets/results/git/bbox-384 /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-GIT-hf/results/384


##### E10 GIT - all - bbox-seg-384 TRAIN 
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-git.sh rahul.mehta@ada:circuitQA/models-hf/ada-script-git-all-bbox-seg-384.sh




##### E2 WCE
cp models-hf/git-files/modeling_git.py $SITE_PACKAGES_PATH/transformers/models/git/modeling_git.py

scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-git.sh rahul.mehta@ada:circuitQA/models-hf/ada-script-git-wce-384.sh


##### E4 WCE
scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-git.sh rahul.mehta@ada:circuitQA/models-hf/ada-script-git-all-384.sh



pip show transformes
SITE_PACKAGES_PATH='/home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages'
cp 


################################################     PIX experiments 
scp datasets/384a.tar rahul.mehta@ada:circuitQA/datasets/

# Get pix-files fodler
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/pix-files/image_processing_pix2struct.py rahul.mehta@ada:circuitQA/models-hf/pix-files
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/pix-files/modeling_pix2struct.py rahul.mehta@ada:circuitQA/models-hf/pix-files


scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/pix-vqa-train-dp.py rahul.mehta@ada:circuitQA/models-hf/
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/pix-vqa-inference.py rahul.mehta@ada:circuitQA/models-hf/

# e1
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-base-384.sh
conda activate cqa-size
sbatch models-hf/pix-base-384.sh

# e1 inference
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/pix-vqa-inference.py rahul.mehta@ada:circuitQA/models-hf/
conda activate cqa-size
sbatch models-hf/infer-pix-base-384.sh


scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/infer-pix-base-384.sh
conda activate cqa-size
sbatch models-hf/infer-pix-base-384.sh 

# e1 save checkpoints
scp -r rahul.mehta@ada:circuitQA/checkpoints-pix-base-384-4Dec/ /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/PIX


# e3
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-post-384.sh
conda activate cqa-size
sbatch models-hf/pix-post-384.sh

scp -r rahul.mehta@ada:circuitQA/checkpoints-pix-post-384-4Dec/ /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/PIX



# e5
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-bbox-384.sh
conda activate cqa-size
sbatch models-hf/pix-bbox-384.sh

scp -r rahul.mehta@ada:circuitQA/checkpoints-pix-bbox-384-4Dec/ /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/PIX
scp -r rahul.mehta@ada:circuitQA/datasets/results/pix/bbox-384/ /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-PIX-hf/results/384

# e5 - infer 
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/infer-pix-bbox-384.sh
conda activate cqa-size
sbatch models-hf/infer-pix-bbox-384.sh


# e6
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-bbox-seg-384.sh
conda activate cqa-size
sbatch models-hf/pix-bbox-seg-384.sh


# e7

scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-ocr-384.sh
conda activate cqa-size
sbatch models-hf/pix-ocr-384.sh 

# e8 
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-desc-384.sh
conda activate cqa-size
sbatch models-hf/pix-desc-384.sh 


# For wce experiments
export SITE_PACKAGES_PATH='/home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages'
cp models-hf/pix-files/modeling_pix2struct.py $SITE_PACKAGES_PATH/transformers/models/pix2struct


# e2 wce
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-pix.sh rahul.mehta@ada:circuitQA/models-hf/pix-wce-384.sh
conda activate cqa-wce
sbatch models-hf/pix-wce-384.sh




