python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=""
nohup accelerate launch finetune_sft.py > output.log 2>&1 &