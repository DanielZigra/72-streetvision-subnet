#!/bin/bash

# Check for required tools
if ! command -v jq &> /dev/null; then
    echo "Please install 'jq' to parse JSON. Exiting."
    exit 1
fi

if ! command -v huggingface-cli &> /dev/null; then
    echo "Please install 'huggingface_hub' and log in with 'huggingface-cli login'. Exiting."
    exit 1
fi

output_file="model_card.json"

# Loop from 11 to 40
for a in $(seq 17 29); do
    hotkey_file="/root/.bittensor/wallets/m-3/hotkeys/bothot${a}"
    
    # Check if file exists
    if [[ ! -f "$hotkey_file" ]]; then
        echo "Skipping hot${a}: file not found."
        continue
    fi

    # Extract ss58Address
    ss58=$(jq -r '.ss58Address' "$hotkey_file")

    if [[ -z "$ss58" || "$ss58" == "null" ]]; then
        echo "Skipping hot${a}: ss58Address not found."
        continue
    fi

    # Get current timestamp
    timestamp=$(date +%s)

    # Create model_card.json
    cat > "$output_file" <<EOF
{
  "model_name": "Zigra/Sam_0${a}",
  "description": "ViT model for roadwork detection.",
  "version": "1.0.0",
  "submitted_by": "${ss58}",
  "submission_time": ${timestamp}
}
EOF
    # Upload model_card.json
    repo_name="Zigra/Sam_0${a}"
    huggingface-cli upload "$repo_name" $output_file
	echo "uploading: ${repo_name}/${output_file}"
    # Create miner.env
    cat > "/root/natix/streetvision-subnet/miner${a}.env" <<EOF
# StreetVision Miner Configuration
#--------------------
# following are initial values
IMAGE_DETECTOR=ViT
IMAGE_DETECTOR_CONFIG=ViT_roadwork.yaml
VIDEO_DETECTOR=TALL
VIDEO_DETECTOR_CONFIG=tall.yaml

# Device Settings
IMAGE_DETECTOR_DEVICE=cpu # Options: cpu, cuda
VIDEO_DETECTOR_DEVICE=cpu

NETUID=72                           # 323 for testnet, 72 for mainnet
SUBTENSOR_NETWORK=finney               # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                     # Endpoints:
                                     # - wss://entrypoint-finney.opentensor.ai:443
                                     # - wss://test.finney.opentensor.ai:443/
                                     
# Wallet Configuration
WALLET_NAME=m-3
WALLET_HOTKEY=bothot${a}

# Miner Settings
MINER_AXON_PORT=70${a}
BLACKLIST_FORCE_VALIDATOR_PERMIT=True # Force validator permit for blacklisting

# Miner details
MODEL_URL=https://huggingface.co/Zigra/Sam_0${a}
PROXY_CLIENT_URL=https://hydra.natix.network
EOF
	# Create start_miner.sh
cat > "/root/natix/streetvision-subnet/start_miner${a}.sh" <<EOF
#!/bin/bash

set -a
source miner${a}.env
set +a

export PYTHONPATH=\$(pwd):\$PYTHONPATH

poetry run python neurons/miner.py \\
  --neuron.image_detector \${IMAGE_DETECTOR:-None} \\
  --neuron.image_detector_config \${IMAGE_DETECTOR_CONFIG:-None} \\
  --neuron.image_detector_device \${IMAGE_DETECTOR_DEVICE:-None} \\
  --netuid \$NETUID \\
  --model_url \$MODEL_URL \\
  --subtensor.network \$SUBTENSOR_NETWORK \\
  --subtensor.chain_endpoint \$SUBTENSOR_CHAIN_ENDPOINT \\
  --wallet.name \$WALLET_NAME \\
  --wallet.hotkey \$WALLET_HOTKEY \\
  --axon.port \$MINER_AXON_PORT \\
  --blacklist.force_validator_permit \$BLACKLIST_FORCE_VALIDATOR_PERMIT \\
  --logging.debug
EOF
	chmod +x /root/natix/streetvision-subnet/start_miner${a}.sh
done