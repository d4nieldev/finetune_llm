# Enter virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Tokens
export HF_TOKEN=""
export WANDB_API_KEY=""

# Run the script in the background
pid = $(nohup accelerate launch finetune_sft.py > output.log 2>&1 &)

echo "Finetuning started with PID: $pid"
echo "To stop the process, use: kill $pid"
echo "To view the logs view output.log"