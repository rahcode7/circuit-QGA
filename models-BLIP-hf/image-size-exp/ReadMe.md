### Experiment - Using 768 size images for fine tuning instead of default 384 

### Update blip transformers package file to resize images to 768

```
pip show transformers # Use the location row that mentions the site packages path

<!-- BLIP_PATH='/Users/rahulmehta/opt/anaconda3/envs/cqa-focal/lib/python3.11/site-packages'  -->

BLIP_PATH=<ENTER SITE PACKAGES PATH >/transformers/models/blip

mv -f BLIP_PATH/modeling_blip.py BLIP_PATH/modeling_blip-original.py
cp -f image-size-exp/modeling_blip.py BLIP_PATH 
```

### Sample file location
https://github.com/rahcode7/circuit-QGA/tree/main/datasets/questions/all



### Run Multi-GPU 
```
cd models-BLIP-hf
sbatch ada_script-ddp-size.sh
```

### Run Interactive Multi-GPU 
```
SCRATCH_DIR='scratch'
CHECKPOINT='checkpoints-ddp'
ROOT_DIR=${PWD}
echo "$ROOT_DIR"


python $ROOT_DIR/models-BLIP-hf/blip-vqa-train-size.py --num_epochs 10 --train_batch_size 8 --val_batch_size 8  --train_dir $ROOT_DIR/datasets  \
    --val_dir $ROOT_DIR/datasets  --checkpoint_dir  $SCRATCH_DIR/CHECKPOINT  \
    --question_dir $ROOT_DIR/datasets/questions/all/master_sample_100.json  --experiment_name ms-blip-single-2 --is_distributed False --wandb_status disabled
```

### Inference on single GPU
```
sbatch ada-script-infer-ddp.sh
```


