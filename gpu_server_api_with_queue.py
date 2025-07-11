import torch
from transformers import ConvNextV2ForImageClassification
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import io
import redis
import hashlib
import uvicorn
from queue import Queue
from threading import Thread, Event
import time

# Redis setup
rdb = redis.Redis(host='localhost', port=6379, db=0)

def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

# Load model once on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ConvNextV2ForImageClassification.from_pretrained("/root/natix/0708/best_model")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def run_inference(image_bytes):
    tensor = preprocess(image_bytes)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output.logits, dim=1)[0,1].item()
        #prob = output.logits.argmax(dim=1).item()
    return prob

# The queue and worker setup
job_queue = Queue()
stop_event = Event()

# Job is a dict: {"image_hash": str, "image_bytes": bytes, "result_event": Event, "result": dict}

def inference_worker():
    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=1)
        except:
            continue  # timeout no job, loop again

        image_hash = job["image_hash"]
        image_bytes = job["image_bytes"]

        # Check cache again (in case result saved meanwhile)
        cached = rdb.get(image_hash)
        if cached:
            prob = float(cached)
        else:
            prob = run_inference(image_bytes)
            rdb.set(image_hash, prob)

        job["result"] = {"probability": prob}
        job["result_event"].set()  # signal result ready
        job_queue.task_done()

worker_thread = Thread(target=inference_worker, daemon=True)
worker_thread.start()

# FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_hash = hash_image_bytes(image_bytes)

    # Check Redis cache first
    cached = rdb.get(image_hash)
    if cached:
        return {"from_cache": True, "probability": float(cached)}

    # Prepare job with an event for signaling
    result_event = Event()
    job = {
        "image_hash": image_hash,
        "image_bytes": image_bytes,
        "result_event": result_event,
        "result": None
    }

    # Put job in queue
    job_queue.put(job)

    # Wait for inference result (blocking here)
    waited = result_event.wait(timeout=60)  # max wait 60s

    if not waited:
        return JSONResponse(content={"error": "Inference timeout"}, status_code=504)

    # Return result
    return {"from_cache": False, "probability": job["result"]["probability"]}

@app.on_event("shutdown")
def shutdown_event():
    stop_event.set()
    worker_thread.join()

if __name__ == "__main__":
    uvicorn.run("gpu_server_api_with_queue:app", host="0.0.0.0", port=8000)
