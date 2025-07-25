# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

import argparse

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import subprocess

import bittensor as bt

from natix.utils.logging import setup_events_logger


def get_device():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT)
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def replace_empty_with_default(args: argparse.Namespace, parser: argparse.ArgumentParser):
    for action in parser._actions:
        arg_name = action.dest
        if isinstance(getattr(args, arg_name), str) and getattr(args, arg_name) == "":
            setattr(args, arg_name, action.default)
    return args


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    config = replace_empty_with_default(config, add_all_args(cls))
    config.logging.info = True

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(config.neuron.full_path, config.neuron.events_retention_size)
        bt.logging.register_primary_logger(events_logger.name)
        if config.logging.debug:
            bt.logging.enable_debug()
        elif config.logging.trace:
            bt.logging.enable_trace()
        elif config.logging.info:
            bt.logging.enable_info()
        else:
            bt.logging.enable_default()


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.image_detector_config",
        type=str,
        help=".yaml file name in base_miner/deepfake_detectors/configs/ to load for trained model.",
        default="roadwork.yaml",
    )

    parser.add_argument(
        "--neuron.image_detector",
        type=str,
        help="The DETECTOR_REGISTRY module name of the DeepfakeDetector subclass to use for inference.",
        default="ROADWORK",
    )

    parser.add_argument(
        "--neuron.image_detector_device",
        type=str,
        help="Device to run image detection model on.",
        default=get_device(),
    )

    parser.add_argument(
        "--neuron.video_detector_config",
        type=str,
        help=".yaml file name in base_miner/deepfake_detectors/configs/ to load for trained model.",
        default="tall.yaml",
    )

    parser.add_argument(
        "--neuron.video_detector",
        type=str,
        help="The DETECTOR_REGISTRY module name of the DeepfakeDetector subclass to use for inference.",
        default="TALL",
    )

    parser.add_argument(
        "--neuron.video_detector_device",
        type=str,
        help="Device to run image detection model on.",
        default=get_device(),
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )

    bt.logging.info("--model_url add argument")
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL to the model submitted on hugging-face.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=get_device(),
    )

    parser.add_argument(
        "--neuron.prompt_type",
        type=str,
        help="Choose 'annotation' to generate prompts from BLIP-2 annotations of real images, or 'random' for arbitrary prompts.",
        default="annotation",
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=10,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=40096,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="template-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="opentensor-dev",
    )

    parser.add_argument(
        "--proxy.port",
        type=int,
        help="The port to run the proxy on.",
        default=10913,
    )

    parser.add_argument(
        "--proxy.proxy_client_url",
        type=str,
        help="The url initialize credentials for proxy.",
        default="https://hydra.dev.natix.network"
    )

    parser.add_argument(
        "--organic.miners_per_task",
        type=int,
        help="Number of miners to query per organic task.",
        default=3,
    )

    parser.add_argument(
        "--organic.deduplication_window_seconds",
        type=int,
        help="Time window in seconds to check for duplicate organic tasks.",
        default=300,
    )

    parser.add_argument(
        "--organic.miner_cooldown_seconds",
        type=int,
        help="Cooldown period in seconds before reassigning similar tasks to the same miner.",
        default=60,
    )

    parser.add_argument(
        "--organic.max_concurrent_tasks",
        type=int,
        help="Maximum number of concurrent organic tasks.",
        default=10,
    )

    parser.add_argument(
        "--organic.stagger_delay_min",
        type=float,
        help="Minimum delay in seconds between staggered miner queries.",
        default=0.1,
    )

    parser.add_argument(
        "--organic.stagger_delay_max",
        type=float,
        help="Maximum delay in seconds between staggered miner queries.",
        default=2.0,
    )


def add_all_args(cls):
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return parser


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = add_all_args(cls)
    return bt.config(parser)
