python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=""
python3 finetune_sft.py