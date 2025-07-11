# The MIT License (MIT)
# Copyright © 2023 Yuma
# developer: dubm
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

import os
import time
import random
import logging

import natix
from natix.validator.cache import ImageCache
from natix.validator.config import (
    ROADWORK_IMAGE_CACHE_DIR,
)
from natix.utils.image_transforms import apply_augmentation_by_level
from transformers import ConvNextV2ForImageClassification, AutoImageProcessor
import sys, termios, tty, time, select
from PIL import Image
import torch
import torchvision.transforms as transforms

TARGET_IMAGE_SIZE: tuple[int, int] = (224, 224)

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all levels, filter will handle output

# Create file handler
handler = logging.FileHandler('my_log.log')
handler.setLevel(logging.DEBUG)

# Add the custom filter
handler.addFilter(InfoOnlyFilter())

# Set format
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

# Clear default handlers and add the custom one
logger.handlers = []
logger.addHandler(handler)

def pass_one_image_all_models(label, level, image, models: list):

    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    if image.mode != 'RGB':
        print(f"Image mode {image.mode} is not RGB, converting to RGB")
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)  # Ensure image is a tensor

    slog = f"Label: {label}, Level: {level}, "       
    
    for i, model in enumerate(models):
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = probabilities[0, 1].item()  # probability of class 1
            if (prediction > 1) or (prediction < 0) or (label == 1 and prediction < 0.5) or (label == 0 and prediction > 0.5):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = challenge["path"]
                output_dir = f"error_images/{label}"
                os.makedirs(output_dir, exist_ok=True)
                filename = filename.split("/")[-1]
                filename = f"{output_dir}/{timestamp}_{filename}"                
                #image = transforms.ToPILImage()(input_tensor.cpu().detach())
                image.save(filename, format='JPEG')
                slog += f"Model {i}: ❌, "
            else:
                slog += f"Model {i}: ✔️, "
            
    slog += challenge["path"]
    #slog = slog[:-2] # Remove trailing tab
    logging.info(slog)

try:
    image_cache = ImageCache(ROADWORK_IMAGE_CACHE_DIR)

    model_paths = [
        "/workspace/testing/01/best_model",
        "/workspace/testing/02/best_model",
        "/workspace/streetvision-subnet/saved_model/checkpoint_epoch_7",
    ]

    if len(model_paths) == 0:
        raise ValueError("No model paths provided. Please specify at least one model path.")
        sys.exit(1)
    
    models = [ConvNextV2ForImageClassification.from_pretrained(p, local_files_only=True) for p in model_paths]

    for model in models:
        model.to(device = device)
        model.eval()
    
    while True:
        label = random.choice([0, 1])
        challenge = image_cache.sample(label)
        if challenge is None:
            print("Waiting for cache to populate. Challenge skipped.")
            continue
        image = challenge["image"]  # extract image
        if image is None:
            print("No image found in challenge, skipping.")
            continue

        image, level, data_aug_params = apply_augmentation_by_level(
            image, TARGET_IMAGE_SIZE, challenge.get("mask_center", None)
        )

        pass_one_image_all_models(label, level, image, models)

        #time.sleep(2)  # Sleep to avoid overwhelming the system
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == 'q':
                print("Exiting...")
                break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)

