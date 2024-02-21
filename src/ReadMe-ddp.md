## BLIP Multi-GPU
squeue -u $USER 


sinteractive -c 20 -g 1
pip install bitsandbytes scipy
pip install transformers==4.31.0
pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113
pip install --upgrade torch torchvision
pip install scipy

squeue -u $USER    

<-- bitsandbytes-0.41.3.post2 -->

conda activate cqa-size

CUDA_VERSION=121 make cuda121

# CUDA VERSION
nvidia-smi
12.1  



## ########     Error

# Soln 1 Copy -higest upvotes
cp /home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda121.so /home2/rahul.mehta/miniconda3/envs/cqa-wce/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so 
cp libbitsandbytes_cuda121.so libbitsandbytes_cpu.so

## Soln 2 
conda install cudatoolkit -y


ln -s /usr/lib/wsl/lib/libcuda.so /home2/rahul.mehta/miniconda3/envs/cqa-size/lib/libcuda.so

conda install -c "nvidia/label/cuda-11.7.1" \
	cuda-libraries=11.7.1 \
	cuda-libraries-dev=11.7.1 \
	cudnn


## Get nvcc backend runtime
conda install -c conda-forge cudatoolkit-dev

# BLIP 
SIZE='576a'
export CHECKPOINT_DIR_BLIP=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/$SIZE-ddp
export CHECKPOINT_DIR_GIT=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/GIT/$SIZE-ddp
export RESULTS_DIR_BLIP=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results-ddp/$SIZE
export RESULTS_DIR_GIT=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-GIT-hf/results-ddp/$SIZE
export LOCAL_SCRIPTS=/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf
export ADA='rahul.mehta@ada:circuitQA'


### Base test 100 or 400 samples
conda activate cqa-base
sbatch models-hf/blip-base-ddp-test.sh        
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-base-ddp-test-100.sh



cat runs/blip/blip-base-ddp-test-24Dec.txt
vi checkpoints-blip-base-ddp-test-100-384-24Dec/statistics.log
ls checkpoints-blip-base-ddp-test-100-384-24Dec


##### Copy train file ddp
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/blip-vqa-train-ddp.py rahul.mehta@ada:circuitQA/models-hf/

##### Copy inference file
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/blip-vqa-inference.py rahul.mehta@ada:circuitQA/models-hf/



#### Base
conda activate cqa-base
sbatch models-hf/blip-base-ddp.sh   

scp $LOCA

cat runs/blip/blip-base-ddp-24Dec.txt
vi checkpoints-blip-base-ddp-24Dec/statistics.log
scp -r rahul.mehta@ada:circuitQA/checkpoints-blip-base-ddp-24Dec /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/384-ddp

####### Inference
sbatch models-hf/blip-base-infer.sh   
cat runs/blip/blip-base-infer.txt
scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/ddp/384a/base /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results-ddp/384a


### Full
### Adamw test WITH DDP 
sbatch models-hf/blip-base-ddp-test-adamw.sh
cat runs/blip/blip-base-ddp-adamw-test-24Dec.txt

## Accelerate test with DDP 
conda activate cqa-base
sbatch models-hf/blip-base-test.sh
cat runs/blip/blip-base-test-ddp-26Dec.txt


### Base + LR Scheduler
conda activate cqa-base
sbatch models-hf/blip-base-lr.sh
cat runs/blip/blip-base-lr.txt

scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-base-lr-infer.sh
sbatch models-hf/blip-base-lr-infer.sh
cat runs/blip/blip-base-lr-infer.txt

scp -r rahul.mehta@ada:circuitQA/checkpoints-blip-ddp-base-lr-27Dec /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/384-ddp 
scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/ddp/384a/base-lr /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results-ddp/384a
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py


### WCE full
scp -r $LOCAL_SCRIPTS/blip-files/configuration_blip.py $ADA/models-hf/blip-files

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-wce.sh $ADA/models-hf
scp -r $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf

sbatch models-hf/blip-wce.sh
cat runs/blip/blip-wce.txt



###### inference


scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-wce-infer.sh
sbatch models-hf/blip-wce-infer.sh
cat runs/blip/blip-wce-infer.txt



### OCR (Pre)
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-ocr-pre.sh

sbatch models-hf/blip-ocr-pre.sh
cat runs/blip/blip-ocr-pre.txt



scp -r rahul.mehta@ada:circuitQA/checkpoints-blip-ddp-ocr-pre-28Dec $CHECKPOINT_DIR
scp -r $CHECKPOINT_DIR_BLIP/checkpoints-blip-ddp-ocr-pre-576-8Jan  $ADA


scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-ocr-pre.sh $ADA/models-hf/blip-ocr-pre-infer.sh
sbatch models-hf/blip-ocr-pre-infer.sh
cat runs/blip/blip-ocr-pre-infer.txt
cat runs/blip/blip-ocr-pre-576-infer.txt

scp -r $ADA/datasets/results/blip/ddp/384a/ocr-pre $RESULTS_DIR
scp -r $ADA/datasets/results/blip/ddp/576a/ocr-pre $RESULTS_DIR_BLIP


python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py

### OCR (Post)
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-ocr-post.sh
sbatch models-hf/blip-ocr-post.sh
cat runs/blip/blip-ocr-post.txt

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-ocr-post.sh $ADA/models-hf/blip-ocr-post-infer.sh 
sbatch models-hf/blip-ocr-post-infer.sh
cat runs/blip/blip-ocr-post-576-infer.txt


scp -r $ADA/datasets/results/blip/ddp/576a/ocr-post $RESULTS_DIR_BLIP

### DESC
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-desc.sh    
sbatch models-hf/blip-desc.sh    
cat runs/blip/blip-desc.txt

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-desc.sh $ADA/models-hf/blip-desc-infer.sh 
sbatch models-hf/blip-desc-infer.sh    
cat runs/blip/blip-desc-576-infer.txt

scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/ddp/384a/desc $RESULTS_DIR
scp -r rahul.mehta@ada:circuitQA/checkpoints-blip-ddp-desc-28Dec $CHECKPOINT_DIR

scp -r $ADA/datasets/results/blip/ddp/576a/ocr-desc $RESULTS_DIR_BLIP


#### BBOX
scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-bbox.sh $ADA/models-hf
sbatch models-hf/blip-bbox.sh    
cat runs/blip/blip-bbox-576.txt

scp -r $ADA/checkpoints-blip-ddp-bbox-28Dec $CHECKPOINT_DIR

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-bbox.sh $ADA/models-hf/blip-bbox-infer.sh
sbatch models-hf/blip-bbox-infer.sh
cat runs/blip/blip-bbox-576-infer.txt


scp -r $ADA/checkpoints-blip-ddp-bbox-29Dec $CHECKPOINT_DIR
scp -r $ADA/checkpoints-blip-ddp-bbox-576-8Jan $CHECKPOINT_DIR_BLIP

scp -r $ADA/datasets/results/blip/ddp/384a/bbox $RESULTS_DIR
scp -r $ADA/datasets/results/blip/ddp/576a/bbox $RESULTS_DIR_BLIP

python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py


#### BBOX SEGMENT

scp -r $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf
scp -r $LOCAL_SCRIPTS/blip-files/configuration_blip.py $ADA/models-hf/blip-files/configuration_blip.py
scp -r $LOCAL_SCRIPTS/blip-files/modeling_blip_text.py $ADA/models-hf/blip-files/modeling_blip_text.py


scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-bbox-segment.sh $ADA/models-hf
sbatch models-hf/blip-bbox-segment.sh    
cat runs/blip/blip-bbox-segment.txt

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-bbox-segment.sh $ADA/models-hf/blip-bbox-segment-infer.sh
sbatch models-hf/blip-bbox-segment-infer.sh
cat runs/blip/blip-bbox-segment-infer.txt
cat runs/blip/blip-bbox-segment-infer-576.txt



scp -r $ADA/checkpoints-blip-ddp-bbox-segment-7Jan $CHECKPOINT_DIR_BLIP
scp -r $CHECKPOINT_DIR_BLIP/checkpoints-blip-ddp-bbox-segment-7Jan  $ADA
scp -r $ADA/datasets/results/blip/ddp/576a/bbox-segment $RESULTS_DIR
scp -r $ADA/runs/blip/blip-bbox-segment.txt $CHECKPOINT_DIR_BLIP

#### ALL experiment 
#####  Base LR + ALL - ocr pre + desc + bbox + bbox segment 
scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/all/master_bbox_and_segment_adv.json $ADA/datasets/questions/all


scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-all.sh $ADA/models-hf/blip-all.sh 
sbatch models-hf/blip-all.sh 
cat runs/blip/blip-all.txt


scp -r $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf
scp -r $LOCAL_SCRIPTS/blip-vqa-inference.py $ADA/models-hf
scp -r $ADA/checkpoints-blip-ddp-all-30Dec $CHECKPOINT_DIR


scp -r $LOCAL_SCRIPTS/ada-scripts/blip/blip-all.sh $ADA/models-hf/blip-all-infer.sh
sbatch models-hf/blip-all-infer.sh 
cat runs/blip/blip-all-infer.txt


scp -r $ADA/datasets/results/blip/ddp/384a/all $RESULTS_DIR
scp -r $ADA/datasets/results/blip/ddp/576a/all $RESULTS_DIR

#### OCR PRE + DESC
scp -r $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf
scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-ocr-pre-desc.sh $ADA/models-hf/blip-ocr-pre-desc.sh 
cat runs/blip/ocr-pre-desc.txt


scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-ocr-pre-desc.sh $ADA/models-hf/blip-ocr-pre-desc-infer.sh
sbatch models-hf/blip-ocr-pre-desc-infer.sh
cat runs/blip/ocr-pre-desc-infer.txt


scp -r $ADA/checkpoints-blip-ddp-ocr-pre-desc-2Jan  $CHECKPOINT_DIR_BLIP
scp -r $ADA/datasets/results/blip/ddp/576a/ocr-pre-desc $RESULTS_DIR_BLIP

<!-- ###### 30 e + all
scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-all.sh $ADA/models-hf/blip-all-30e.sh 
sbatch models-hf/blip-all-30e.sh  -->


##  576 EXPeriments

#### BASE

SITE_PACKAGES_PATH='/home2/rahul.mehta/anaconda3/envs/cqa-base/lib/python3.8/site-packages'
vi $SITE_PACKAGES_PATH/transformers/models/blip/modeling_blip.py

scp $LOCAL_SCRIPTS/blip-vqa-train-ddp.py  $ADA/models-hf
scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-base.sh $ADA/models-hf/blip-base-576.sh
scp $LOCAL_SCRIPTS/blip-files/modeling_blip_384.py $ADA/models-hf/blip-files
scp $LOCAL_SCRIPTS/blip-files/modeling_blip_576.py $ADA/models-hf/blip-files
scp $LOCAL_SCRIPTS/blip-files/original/modeling_blip_text.py $ADA/models-hf/blip-files/original
scp $LOCAL_SCRIPTS/blip-files/original/configuration_blip.py $ADA/models-hf/blip-files/original

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-base.sh $ADA/models-hf/blip-base-576.sh
scp $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf/



conda activate cqa-base
sbatch models-hf/blip-base-576.sh


scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-base.sh $ADA/models-hf/blip-base-576-infer.sh
sbatch models-hf/blip-base-576-infer.sh
cat runs/blip/blip-base-576-infer.txt


scp -r $ADA/checkpoints-blip-ddp-base-576-6Jan $CHECKPOINT_DIR_BLIP  
scp -r $ADA/datasets/results/blip/ddp/576a/base-576 $RESULTS_DIR_BLIP

python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py


##### Base + LR Scheduler

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-base-lr.sh $ADA/models-hf/blip-base-lr-576.sh
scp $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf/


conda activate cqa-base
sbatch models-hf/blip-base-lr-576.sh
cat runs/blip/blip-base-lr-576.txt

scp /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/ada-script-blip.sh rahul.mehta@ada:circuitQA/models-hf/blip-base-lr-infer.sh
sbatch models-hf/blip-base-lr-infer.sh
cat runs/blip/blip-base-lr-infer.txt

scp -r rahul.mehta@ada:circuitQA/checkpoints-blip-ddp-base-lr-27Dec /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/checkpoints/BLIP/384-ddp 
scp -r rahul.mehta@ada:circuitQA/datasets/results/blip/ddp/384a/base-lr /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-BLIP-hf/results-ddp/384a
python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py

scp -r $CHECKPOINT_DIR_BLIP/checkpoints-blip-ddp-base-lr-576-6Jan   $ADA
scp -r $ADA/datasets/results/blip/ddp/576a/base-lr $RESULTS_DIR_BLIP

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-base.sh $ADA/models-hf/blip-base-lr-infer.sh
sbatch models-hf/blip-base-lr-infer.sh
cat runs/blip/blip-base-lr-infer.txt



python /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/evaluate/evaluate-pred.py


###### WCE
scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-wce.sh $ADA/models-hf/blip-wce.sh
scp $LOCAL_SCRIPTS/blip-vqa-train-ddp.py $ADA/models-hf/

conda activate cqa-wce

sbatch models-hf/blip-wce-576.sh
cat runs/blip/blip-wce-576.txt

scp $LOCAL_SCRIPTS/ada-scripts/blip/blip-wce.sh $ADA/models-hf/blip-wce-infer.sh
sbatch models-hf/blip-wce-infer.sh
cat runs/blip/blip-wce-576-infer.txt


scp -r $LOCAL_SCRIPTS/blip-vqa-inference.py $ADA/models-hf
scp -r $ADA/checkpoints-blip-ddp-wce-5Jan  $CHECKPOINT_DIR_BLIP
scp -r $ADA/datasets/results/blip/ddp/576a/wce $RESULTS_DIR_BLIP

