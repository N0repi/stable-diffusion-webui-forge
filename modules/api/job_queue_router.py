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
        return self.job_status.get(job_id, {"status": "pending", "message": "Job not started yet"})

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
        response = requests.post(endpoint, json=payload, timeout=420)

        if response.status_code == 200:
            result = response.json()
            job_queue.update_status(job_id, "completed", result=result)
            print(f"✅ Job {job_id} completed in {time.time() - t0:.2f}s")
        else:
            raise ValueError(f"Forge returned {response.status_code}: {response.text[:300]}")

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        job_queue.update_status(job_id, "failed", error=str(e))

