# job_queue_router.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import uuid
import requests
import time
import asyncio
from threading import RLock

# Import our Redis-based job queue
from .redis_job_queue import job_queue

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

@router.get("/")
def root():
    return {"message": "Online"}

# ✅ Launch background image processing
@router.post("/generate")
async def generate_image(request: ImageGenRequest, background_tasks: BackgroundTasks):
    # Convert request to dict for Redis storage
    request_data = {
        "prompt": request.prompt,
        "seed": request.seed,
        "init_image": request.init_image,
        "mode": request.mode,
        "api_url": request.api_url,
        "denoising_strength": request.denoising_strength
    }
    
    job_id = await job_queue.create_job(request_data)
    print(f"🧵 Job {job_id} queued in Redis")
    return {"message": "Job queued", "job_id": job_id}

# ✅ Poll job status
@router.get("/generate/status/{job_id}")
async def get_job_status(job_id: str):
    return await job_queue.get_job_status(job_id)

# 🔧 Background processing is now handled by Redis worker
# The worker runs separately and processes jobs from the Redis queue
# See redis_worker.py for the processing logic


