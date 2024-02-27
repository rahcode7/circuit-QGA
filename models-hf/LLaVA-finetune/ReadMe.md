ssh rahul.mehta@ada
sinteractive -c 20 -g 1
conda activate cqa-base
pip install deepspeed



scp -r /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/models-hf/LLaVA-finetune/ada-script-llava.sh rahul.mehta@ada:circuitQA/models-hf/