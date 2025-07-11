# The MIT License (MIT)
# Copyright © 2023 Yuma
# Copyright © 2023 Natix

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import base64
import io
import time
import typing

import bittensor as bt
from PIL import Image

from base_miner.detectors import RoadworkDetector, ViTImageDetector  # noqa: F401
from base_miner.registry import DETECTOR_REGISTRY
from natix.base.miner import BaseMinerNeuron
from natix.protocol import ExtendedImageSynapse
from natix.utils.config import get_device

import logging
import os
import torch
import requests
import redis
import hashlib
from requests.exceptions import Timeout

rdb = redis.Redis(host='localhost', port=6379, db=0)

def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def predict_with_cache(image_bytes: bytes, timeout_sec=3, max_retries=3) -> float:
    image_hash = hash_image_bytes(image_bytes)

    # Check Redis cache first
    cached = rdb.get(image_hash)
    if cached:
        print("🟡 Found in local Redis cache.")
        return float(cached)

    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post("http://localhost:8000/predict", files=files, timeout=timeout_sec)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                print(f"❌ Server error: {result['error']}")
                continue

            if not result["from_cache"]:
                rdb.set(image_hash, result["probability"])
                print("🟢 Inference from GPU.")
            else:
                print("🟢 Served from server cache.")

            return result["probability"]

        except Timeout:
            print(f"⚠️ Timeout on attempt {attempt}/{max_retries}. Retrying...")
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            break

    return 0.999999999

class Miner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_image,
            blacklist_fn=self.blacklist_image,
            priority_fn=self.priority_image,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        bt.logging.info("Loading image detection model if configured")
        self.load_image_detector()
        
        # Initialize request tracking
        self.request_stats = {
            'total_requests': 0,
            'requests_by_hotkey': {},
            'requests_by_uid': {},
            'requests_by_ip': {},
            'start_time': time.time()
        }
        '''
        # Example: get the UID (replace with actual miner ID source)
        miner_id = self.uid if hasattr(self, "uid") else "unknown"

        # Construct file name with miner ID
        log_filename = f"miner-{miner_id}.log"

        # Set up file logging
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        bt.logging.set_debug(True)
        bt.logging._logger.addHandler(file_handler)
        bt.logging._logger.setLevel(logging.DEBUG)        
        '''

    def load_image_detector(self):
        if (
            str(self.config.neuron.image_detector).lower() == "none"
            or str(self.config.neuron.image_detector_config).lower() == "none"
        ):
            bt.logging.warning("No image detector configuration provided, skipping.")
            self.image_detector = None
            return

        if self.config.neuron.image_detector_device == "auto":
            bt.logging.warning("Automatic device configuration enabled for image detector")
            self.config.neuron.image_detector_device = get_device()

        self.image_detector = DETECTOR_REGISTRY[self.config.neuron.image_detector](
            config_name=self.config.neuron.image_detector_config, device=self.config.neuron.image_detector_device
        )
        bt.logging.info(f"Loaded image detection model: {self.config.neuron.image_detector}")

    async def forward_image(self, synapse: ExtendedImageSynapse) -> ExtendedImageSynapse:
        """
        Perform inference on image

        Args:
            synapse (bt.Synapse): The synapse object containing the list of b64 encoded images in the
            'images' field.

        Returns:
            bt.Synapse: The synapse object with the 'predictions' field populated with a list of probabilities

        """
        # Log sender information
        self._log_sender_info(synapse)
        
        if self.image_detector is None:
            bt.logging.info("Image detection model not configured; skipping image challenge")
        else:
            bt.logging.info("Received image challenge!")
            try:
                image_bytes = base64.b64decode(synapse.image)
                synapse.prediction = predict_with_cache(image_bytes, timeout_sec=4, max_retries=2)
                synapse.model_url = str(self.config.model_url)

            except Exception as e:
                bt.logging.error("Error performing inference")
                bt.logging.error(e)

            bt.logging.info(f"PREDICTION = {synapse.prediction}")
            label = synapse.testnet_label
            if synapse.testnet_label != -1:
                bt.logging.info(f"LABEL (testnet only) = {label}")
        return synapse

    def _log_sender_info(self, synapse: ExtendedImageSynapse):
        """
        Log comprehensive information about the request sender.
        
        Args:
            synapse: The synapse object containing sender information
        """
        try:
            # Get sender hotkey
            sender_hotkey = synapse.dendrite.hotkey if synapse.dendrite else "Unknown"
            
            # Get sender IP address
            sender_ip = synapse.dendrite.ip if synapse.dendrite else "Unknown"
            
            # Get sender UID from metagraph
            sender_uid = None
            sender_stake = 0.0
            sender_trust = 0.0
            sender_incentive = 0.0
            sender_emission = 0.0
            is_validator = False
            
            if sender_hotkey != "Unknown" and sender_hotkey in self.metagraph.hotkeys:
                sender_uid = self.metagraph.hotkeys.index(sender_hotkey)
                sender_stake = float(self.metagraph.S[sender_uid])
                sender_trust = float(self.metagraph.T[sender_uid])
                sender_incentive = float(self.metagraph.I[sender_uid])
                sender_emission = float(self.metagraph.E[sender_uid])
                is_validator = bool(self.metagraph.validator_permit[sender_uid])
            
            # Update request statistics
            self._update_request_stats(sender_hotkey, sender_uid, sender_ip)
            
            # Log comprehensive sender information
            bt.logging.info("=" * 80)
            bt.logging.info("📨 REQUEST RECEIVED - SENDER INFORMATION")
            bt.logging.info("=" * 80)
            bt.logging.info(f"🔑 Hotkey: {sender_hotkey}")
            bt.logging.info(f"🌐 IP Address: {sender_ip}")
            bt.logging.info(f"🆔 UID: {sender_uid if sender_uid is not None else 'Not Registered'}")
            bt.logging.info(f"💰 Stake: {sender_stake:.6f} τ")
            bt.logging.info(f"🤝 Trust: {sender_trust:.6f}")
            bt.logging.info(f"🎯 Incentive: {sender_incentive:.6f}")
            bt.logging.info(f"📈 Emission: {sender_emission:.6f}")
            bt.logging.info(f"✅ Validator: {'Yes' if is_validator else 'No'}")
            bt.logging.info(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            bt.logging.info("=" * 80)
            
            # Additional logging for specific cases
            if sender_hotkey == "Unknown":
                bt.logging.warning("⚠️  Request from unknown sender - no hotkey provided")
            elif sender_uid is None:
                bt.logging.warning(f"⚠️  Hotkey {sender_hotkey} not found in metagraph")
            elif sender_stake == 0.0:
                bt.logging.warning(f"⚠️  Zero stake sender: {sender_hotkey}")
            elif is_validator:
                bt.logging.info(f"✅ Validator request from {sender_hotkey} (UID: {sender_uid})")
            else:
                bt.logging.info(f"📝 Non-validator request from {sender_hotkey} (UID: {sender_uid})")
                
        except Exception as e:
            bt.logging.error(f"Error logging sender information: {e}")
            bt.logging.error(f"Synapse dendrite: {synapse.dendrite}")

    def _update_request_stats(self, hotkey: str, uid: int, ip: str):
        """
        Update request statistics tracking.
        
        Args:
            hotkey: Sender hotkey
            uid: Sender UID
            ip: Sender IP address
        """
        # Update total requests
        self.request_stats['total_requests'] += 1
        
        # Update requests by hotkey
        if hotkey not in self.request_stats['requests_by_hotkey']:
            self.request_stats['requests_by_hotkey'][hotkey] = 0
        self.request_stats['requests_by_hotkey'][hotkey] += 1
        
        # Update requests by UID
        uid_key = str(uid) if uid is not None else "Unknown"
        if uid_key not in self.request_stats['requests_by_uid']:
            self.request_stats['requests_by_uid'][uid_key] = 0
        self.request_stats['requests_by_uid'][uid_key] += 1
        
        # Update requests by IP
        if ip not in self.request_stats['requests_by_ip']:
            self.request_stats['requests_by_ip'][ip] = 0
        self.request_stats['requests_by_ip'][ip] += 1

    def get_request_stats(self):
        """
        Get current request statistics.
        
        Returns:
            dict: Current request statistics
        """
        uptime = time.time() - self.request_stats['start_time']
        stats = self.request_stats.copy()
        stats['uptime_seconds'] = uptime
        stats['uptime_hours'] = uptime / 3600
        stats['requests_per_hour'] = (stats['total_requests'] / stats['uptime_hours']) if stats['uptime_hours'] > 0 else 0
        
        return stats

    def export_request_stats(self, filename: str = None):
        """
        Export request statistics to a JSON file.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"request_stats_{timestamp}.json"
        
        stats = self.get_request_stats()
        
        # Convert to JSON-serializable format
        export_stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': stats['uptime_seconds'],
            'uptime_hours': stats['uptime_hours'],
            'total_requests': stats['total_requests'],
            'requests_per_hour': stats['requests_per_hour'],
            'unique_hotkeys': len(stats['requests_by_hotkey']),
            'unique_uids': len(stats['requests_by_uid']),
            'unique_ips': len(stats['requests_by_ip']),
            'requests_by_hotkey': stats['requests_by_hotkey'],
            'requests_by_uid': stats['requests_by_uid'],
            'requests_by_ip': stats['requests_by_ip']
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_stats, f, indent=2)
            bt.logging.info(f"📄 Request statistics exported to: {filename}")
        except Exception as e:
            bt.logging.error(f"Error exporting statistics: {e}")

    def log_request_stats(self):
        """
        Log current request statistics.
        """
        stats = self.get_request_stats()
        
        bt.logging.info("=" * 80)
        bt.logging.info("📊 REQUEST STATISTICS")
        bt.logging.info("=" * 80)
        bt.logging.info(f"⏱️  Uptime: {stats['uptime_hours']:.2f} hours")
        bt.logging.info(f"📈 Total Requests: {stats['total_requests']}")
        bt.logging.info(f"🚀 Requests/Hour: {stats['requests_per_hour']:.2f}")
        bt.logging.info(f"👥 Unique Hotkeys: {len(stats['requests_by_hotkey'])}")
        bt.logging.info(f"🆔 Unique UIDs: {len(stats['requests_by_uid'])}")
        bt.logging.info(f"🌐 Unique IPs: {len(stats['requests_by_ip'])}")
        
        # Top 5 senders by hotkey
        if stats['requests_by_hotkey']:
            top_senders = sorted(stats['requests_by_hotkey'].items(), key=lambda x: x[1], reverse=True)[:5]
            bt.logging.info("🏆 Top 5 Senders (by hotkey):")
            for hotkey, count in top_senders:
                bt.logging.info(f"   {hotkey}: {count} requests")
        
        bt.logging.info("=" * 80)

    async def blacklist_image(self, synapse: ExtendedImageSynapse) -> typing.Tuple[bool, str]:
        # Log all incoming requests for blacklist check
        self._log_blacklist_check(synapse)
        return await self.blacklist(synapse)

    async def priority_image(self, synapse: ExtendedImageSynapse) -> float:
        # Log priority assignment
        self._log_priority_assignment(synapse)
        return await self.priority(synapse)

    def _log_blacklist_check(self, synapse: ExtendedImageSynapse):
        """
        Log information about requests being checked for blacklisting.
        
        Args:
            synapse: The synapse object containing sender information
        """
        try:
            sender_hotkey = synapse.dendrite.hotkey if synapse.dendrite else "Unknown"
            sender_ip = synapse.dendrite.ip if synapse.dendrite else "Unknown"
            
            bt.logging.debug(f"🔍 BLACKLIST CHECK - Hotkey: {sender_hotkey} | IP: {sender_ip}")
            
        except Exception as e:
            bt.logging.error(f"Error in blacklist logging: {e}")

    def _log_priority_assignment(self, synapse: ExtendedImageSynapse):
        """
        Log priority assignment for requests.
        
        Args:
            synapse: The synapse object containing sender information
        """
        try:
            sender_hotkey = synapse.dendrite.hotkey if synapse.dendrite else "Unknown"
            
            if sender_hotkey != "Unknown" and sender_hotkey in self.metagraph.hotkeys:
                sender_uid = self.metagraph.hotkeys.index(sender_hotkey)
                sender_stake = float(self.metagraph.S[sender_uid])
                bt.logging.debug(f"⚡ PRIORITY ASSIGNED - Hotkey: {sender_hotkey} | UID: {sender_uid} | Stake: {sender_stake:.6f} τ")
            else:
                bt.logging.debug(f"⚡ PRIORITY ASSIGNED - Hotkey: {sender_hotkey} | UID: Not Registered")
                
        except Exception as e:
            bt.logging.error(f"Error in priority logging: {e}")

    def save_state(self):
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    with Miner() as miner:
        last_stats_time = time.time()
        while True:
            log = (
                "Miner | "
                f"UID:{miner.uid} | "
                # f"Block:{self.current_block} | "
                f"Stake:{miner.metagraph.S[miner.uid]:.3f} | "
                f"Trust:{miner.metagraph.T[miner.uid]:.3f} | "
                f"Incentive:{miner.metagraph.I[miner.uid]:.3f} | "
                f"Emission:{miner.metagraph.E[miner.uid]:.3f}"
            )
            bt.logging.info(log)
            
            # Log request statistics every 60 seconds
            current_time = time.time()
            if current_time - last_stats_time >= 60:
                miner.log_request_stats()
                last_stats_time = current_time
            
            time.sleep(5)
