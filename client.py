import requests
import redis
import hashlib
from requests.exceptions import Timeout
from PIL import Image
import io
import os

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
            response = requests.post("http://65.109.75.58:8000/predict", files=files, timeout=timeout_sec)
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

def scan_jpeg_files_recursive(folder_path):
    jpeg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                jpeg_files.append(full_path)
    return jpeg_files

# Example usage
if __name__ == "__main__":

    folder = '/root/streetvision-subnet/error_images'
    jpeg_list = scan_jpeg_files_recursive(folder)
    for file_path in jpeg_list:
        print(file_path)

        image = Image.open(file_path).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        prob = predict_with_cache(image_bytes, timeout_sec=60, max_retries=5)
        print(f"🔍 Final prediction: {prob}")
