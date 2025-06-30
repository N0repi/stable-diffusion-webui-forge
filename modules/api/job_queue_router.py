# job_queue_router.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import uuid
import requests
import time
from threading import RLock

# Job Queue Class
class JobQueue:
    def __init__(self):
        self.job_status: Dict[str, Dict] = {}
        self.lock = RLock()

    def create_job(self, request):
        job_id = str(uuid.uuid4())
        self.job_status[job_id] = {"status": "queued", "result": None, "error": None}
        return job_id

    def update_status(self, job_id, status, result=None, error=None):
        with self.lock:
            if job_id in self.job_status:
                self.job_status[job_id]["status"] = status
                self.job_status[job_id]["result"] = result
                self.job_status[job_id]["error"] = error

    def get_status(self, job_id):
        with self.lock:
            if job_id in self.job_status:
                return self.job_status[job_id]
        raise HTTPException(status_code=404, detail="Job not found")

# Request Model
class ImageGenRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    init_image: Optional[str] = None
    mode: str  # "txt2img" or "img2img"
    api_url: str
    denoising_strength: Optional[float] = Field(None, alias="denoisingStrength")

# Router Setup
router = APIRouter()
job_queue = JobQueue()

@router.get("/")
def root():
    return {"message": "Online"}

# ✅ Launch background image processing
@router.post("/generate")
def generate_image(request: ImageGenRequest, background_tasks: BackgroundTasks):
    job_id = job_queue.create_job(request)
    background_tasks.add_task(process_image, request, job_id)
    print(f"🧵 Job {job_id} queued and detached via BackgroundTasks")
    return {"message": "Job queued", "job_id": job_id}

# ✅ Poll job status
@router.get("/generate/status/{job_id}")
def get_job_status(job_id: str):
    return job_queue.get_status(job_id)

# 🔧 Background processing logic
def process_image(request: ImageGenRequest, job_id: str):
    try:
        print(f"🟢 Starting process_image for job_id={job_id}")
        t0 = time.time()

        payload = {
            "prompt": request.prompt,
            "steps": 15,
            "seed": request.seed or uuid.uuid4().int % (2**32),
            "sampler_name": "Euler",
            "width": 768,
            "height": 768,
            "batch_size": 1,
            "cfg_scale": 1,
            "distilled_cfg_scale": 3.5,
            "scheduler": "Simple",
            "override_settings": {
                "forge_additional_modules": [
                    "ae.safetensors",
                    "clip_l.safetensors",
                    "t5xxl_fp16.safetensors",
                ],
                "sd_model_checkpoint": "flux/flux1-dev-fp8.safetensors",
                "forge_unet_storage_dtype": "Automatic",
            },
        }

        if request.mode == "txt2img":
            endpoint = f"{request.api_url}/sdapi/v1/txt2img"
        elif request.mode == "img2img":
            if not request.init_image:
                raise ValueError("init_image is required for img2img mode")
            payload["init_images"] = [request.init_image]
            payload["denoising_strength"] = request.denoising_strength or 0.85
            endpoint = f"{request.api_url}/sdapi/v1/img2img"
        else:
            raise ValueError("Invalid mode")

        print(f"📤 Sending generation request to: {endpoint}")
        MAX_RETRIES = 3
        RETRY_DELAY = 3


        for attempt in range(MAX_RETRIES):
            post_start = time.time()
            try:
                response = requests.post(endpoint, json=payload, timeout=420)
            except Exception as post_err:
                print(f"⚠️ Request attempt {attempt + 1} failed: {post_err}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise

            post_duration = time.time() - post_start
            print(f"📨 Attempt {attempt + 1}: POST completed in {post_duration:.2f}s")

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"✅ JSON parsed successfully on attempt {attempt + 1}")
                    job_queue.update_status(job_id, "completed", result=result)
                    total = time.time() - t0
                    print(f"🏁 Job {job_id} completed successfully in {total:.2f}s")
                    return
                except Exception as json_err:
                    print(f"❌ Invalid JSON on attempt {attempt + 1}: {json_err}")
                    if attempt < MAX_RETRIES - 1:
                        print("🔁 Retrying due to invalid JSON...")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        raise ValueError("Forge returned non-JSON response.")
            elif response.status_code == 404:
                print(f"⚠️ Forge not ready (404) on attempt {attempt + 1}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"❌ Forge returned status {response.status_code} on attempt {attempt + 1}")
                print(f"↩ Response text: {response.text[:300]}...")
                raise ValueError(f"Image generation failed: {response.text}")

        # If we reach here, all attempts failed
        raise RuntimeError("All retry attempts failed for Forge inference")

    except Exception as e:
        print(f"❌ process_image failed for job_id {job_id}: {e}")
        job_queue.update_status(job_id, "failed", error=str(e))


