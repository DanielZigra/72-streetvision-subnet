# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Data 📊](#data)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)

## Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml).

## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/natixnetwork/natix-subnet.git && cd natix-subnet
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can use poetry to setup the dependencies.

```bash
poetry env use path/to/your/python.3.11
poetry sync
```

To activate your virtual environment, run `poetry env activate`.

Install the remaining necessary requirements with the following chained command.

```bash
poetry env activate
chmod +x setup_env.sh
./setup_env.sh
```

If WanDB is activated, you'll need to login first

```
wandb login --relogin
```

Next you need to login with your huggingface credentials

```
huggingface-cli login
```

## Registration

To validate on our subnet, you must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid 34 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 168 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Validating

You can launch your validator with `run_neuron.py`.

First, make sure to update `validator.env` with your **wallet**, **hotkey**, and **validator port**. This file was created for you during setup, and is not tracked by git.

```bash
NETUID=34                                      # Network User ID options: 34, 168
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                                # Endpoints:
                                                # - wss://entrypoint-finney.opentensor.ai:443
                                                # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Note: If you're using RunPod, you must select a port >= 70000 for symmetric mapping
# Validator Port Setting:
VALIDATOR_AXON_PORT=8092
VALIDATOR_PROXY_PORT=10913
DEVICE=cuda

# API Keys:
WANDB_API_KEY=your_wandb_api_key_here
HUGGING_FACE_TOKEN=your_hugging_face_token_here
```

If you don't have a W&B API key, please reach out to the Natix team via Discord and we can provide one.

Now you're ready to run your validator!

```bash
conda activate natix
pm2 start run_neuron.py -- --validator 
```
- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are enabled by default (every 6 hours). To disable, run with `--no-self-heal`.


The above command will kick off 4 `pm2` processes
```
┌────┬───────────────────────────┬─────────────┬─────────┬─────────┬──────────┬────────┬──────┬───────────┬──────────┬──────────┬──────────┬──────────┐
│ id │ name                      │ namespace   │ version │ mode    │ pid      │ uptime │ ↺    │ status    │ cpu      │ mem      │ user     │ watching │
├────┼───────────────────────────┼─────────────┼─────────┼─────────┼──────────┼────────┼──────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│ 2  │ natix_cache_updater     │ default     │ N/A     │ fork    │ 1601308  │ 2h     │ 0    │ online    │ 0%       │ 843.6mb  │ user     │ disabled │
│ 3  │ natix_data_generator    │ default     │ N/A     │ fork    │ 1601426  │ 2h     │ 0    │ online    │ 0%       │ 11.3gb   │ user     │ disabled │
│ 1  │ natix_validator         │ default     │ N/A     │ fork    │ 1601246  │ 2h     │ 0    │ online    │ 0%       │ 867.8mb  │ user     │ disabled │
│ 0  │ run_neuron                │ default     │ N/A     │ fork    │ 223218   │ 41h    │ 0    │ online    │ 0%       │ 8.9mb    │ user     │ disabled │
└────┴───────────────────────────┴─────────────┴─────────┴─────────┴──────────┴────────┴──────┴───────────┴──────────┴──────────┴──────────┴──────────┘
```
- `run_neuron` manages self heals and auto updates
- `natix_validator` is the validator process, whose hotkey, port, etc. are configured in `validator.env`
- `natix_data_generator` runs our data generation pipeline to produce **synthetic images** used for evaluation of miners (stored in `~/.cache/sn34/synthetic`)
- `natix_cache_updater` manages the cache of **real images**  (stored in `~/.cache/sn34/real`) 
