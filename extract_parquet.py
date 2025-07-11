from natix.validator.cache.extract import extract_images_from_parquet
from natix.validator.cache.download import download_files, list_hf_files
from pathlib import Path
import os
import json
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
#import wget
import threading
from natix.validator.config import TARGET_IMAGE_SIZE
from natix.utils.image_transforms import (
    get_base_transforms,
    get_random_augmentations,
    get_random_augmentations_medium,
    get_random_augmentations_hard
)

HUGGINGFACE_REPO = os.getenv("HUGGINGFACE_REPO", "natix-network-org")

DATASETS = {
#    "None": f"{HUGGINGFACE_REPO}/none",
    "Roadwork": f"{HUGGINGFACE_REPO}/roadwork"
}

root_dir = "/root/.cache/natix" #"/root/natix-ds"
'''
# Download parquet Files
for dataset in DATASETS:
    filenames = list_hf_files(repo_id=DATASETS[dataset], extension=".parquet")
    compressed_dir = f"{root_dir}/{dataset}/image/sources"
    for f in filenames:
        path = f"https://huggingface.co/datasets/{DATASETS[dataset]}/resolve/main/{f}"
        print(f"Downloading {path} to {compressed_dir}")
        download_files(urls={path}, output_dir=compressed_dir)
#wget.download(url="https://huggingface.co/datasets/Zigra/Data/resolve/main/data-00001-of-00001.parquet?download=true", out=f"{root_dir}/None/image/sources/data-00001-of-00001.parquet")

# Extarct images from parquet files
for dataset in DATASETS:
    compressed_dir = f"{root_dir}/{dataset}/image/sources"
    extract_dir = f"{root_dir}/{dataset}/image"
    filenames = os.listdir(compressed_dir)
    for f in filenames:
        if f == 'data-00000-of-00001.parquet':
            continue
        parquet_path = f"{compressed_dir}/{f}"
        print(f"Extracting images from {parquet_path} to {extract_dir}")
        extract_images_from_parquet(Path(parquet_path), extract_dir, 1000000)
'''
def augument_data(f: str, extract_dir : str):
    image_path = f"{extract_dir}/{f}"
    file_title = os.path.splitext(f)[0]
    json_path = f"{extract_dir}/{file_title}.json"
    print(f"Processing {image_path} with metadata {json_path}")
    if not os.path.exists(json_path):
        print(f"Warning: No JSON metadata found for {image_path}. Skipping.")
        return
    metadata = json.loads(Path(json_path).read_text())
    label = metadata.get("label", None)
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    if (label == 1):
        return
    if (metadata.get("scene_description") and label == 0) or (not metadata.get("scene_description") and label == 1):
        print(f"Warning: Scene description mismatch for {image_path}. Skipping.")
        return
    if label is None or isinstance(label, str):
        print(f"Warning: No label found in metadata for {image_path}. Skipping.")
    else:
        target_dir = f"{root_dir}/totrain/{label}/{file_title}"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        else:
            return
        image = Image.open(image_path)
        tforms = {f"0_0": get_base_transforms(TARGET_IMAGE_SIZE)}
        mask_point = metadata.get(f"mask_center", None)
        for i in range(380 if label == 0 else 50):
        #for i in range(30):
            for level in range(1, 4):
                if level == 1:
                    tforms |= {f"{level}_{i:03}": get_random_augmentations(TARGET_IMAGE_SIZE, mask_point)}
                elif level == 2:
                    tforms |= {f"{level}_{i:03}": get_random_augmentations_medium(TARGET_IMAGE_SIZE, mask_point)}
                else:  # level == 3
                    tforms |= {f"{level}_{i:03}": get_random_augmentations_hard(TARGET_IMAGE_SIZE, mask_point)}

        for key, tform in tforms.items():
            target_path = os.path.join(target_dir, f"{key}.jpg")
            transformed = tform(image)
            to_pil = T.ToPILImage()
            img = to_pil(transformed)  # Ensure tensor is on CPU
            img.save(target_path)

for dataset in DATASETS:
    extract_dir = f"{root_dir}/{dataset}/image"
    filenames = os.listdir(extract_dir)

    threads = []
    batch_size = 4

    for f in filenames:
        if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.mpo'):
            #augument_data(f, extract_dir)
            t = threading.Thread(target=augument_data, kwargs={'f': f, 'extract_dir': extract_dir})
            threads.append(t)
            t.start()

            # When batch is full, wait for all to finish
            if len(threads) >= batch_size:
                for t in threads:
                    t.join()
                threads = []  # Reset for next batch
    
    # Join any remaining threads
    for t in threads:
        t.join()

#extract_images_from_parquet(Path('/root/.cache/natix/None/image/sources/data-00000-of-00001.parquet'), '/root/.cache/natix/None/image', 0)