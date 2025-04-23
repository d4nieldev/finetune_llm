python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=""
export WANDB_API_KEY=""
pid = $(nohup accelerate launch finetune_sft.py > output.log 2>&1 &)
echo "Finetuning started with PID: $pid"
echo "To stop the process, use: kill $pid"
echo "To view the logs, use: tail -f output.log"
echo "To view the process list, use: ps aux | grep finetune_sft.py"