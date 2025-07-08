# Copy this script to the remote machine, e.g.
# scp ./run_on_gpu.sh root@[IP]:/root/run_on_gpu.sh
# Then ssh into the remote and run this script

apt-get update
apt-get install git
#apt-get install git-lfs
apt-get install tmux -y
apt-get install nvtop -y
# Use /workspace if available (much more disk space in runpod containers)

git clone https://github.com/nick-kuhn/mlx_week5_gigglers
cd mlx_week5_gigglers

# Install uv after we're in the target directory to avoid path issues
# git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv sync

# Change to suit your needs! 
git config --global user.email "example@user.com"
git config --global user.name "my-github-username"

# You can generate a new token at https://github.com/settings/personal-access-tokens
# => Select only this repository
# => Select Read and Write access to Contents (AKA Code)

# Optional: Login to Hugging Face with access token
#echo "Logging into Hugging Face..."
#uv run huggingface-cli login 

# Login to Weights and Biases
uv run wandb login 

# Launches a new tmux session (with name sweep) the name is optional!
# This session can survive even if you disconnect from SSH
# => Ctrl+B enters command mode in tmux (then release ctrl)
# ==> Ctrl+B (unclick Ctrl) then D detaches from the current tmux session
# => Discover existing sessions with tmux ls
# => Reattach to the last session with tmux a (short for attach)
# => Reattach with tmux attach -t 0
# => Scroll with Ctrl+B [ then use the arrow keys or mouse to scroll up and down. Leave scroll mode with Esc or q

tmux new -s mlx

# Check GPU usage with the nvtop command

# Now run a script, e.g.
# uv run -m model.continue_train
#uv run wandb_sweep.py