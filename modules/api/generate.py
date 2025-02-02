# generate.py

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import uuid
import requests

# Create a router instead of a standalone app
router = APIRouter()

# In-memory job status store
job_status: Dict[str, Dict] = {}

# Request body model for txt2img and img2img
class ImageGenRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    init_image: Optional[str] = None  # Only for img2img
    mode: str  # "txt2img" or "img2img"
    api_url: str  # URL for the API endpoint (dynamic)
    denoising_strength: Optional[float] = Field(None, alias="denoisingStrength")

@router.post("/generate")
def generate_image(request: ImageGenRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "queued", "result": None, "error": None}

    # Background task to process the image
    background_tasks.add_task(process_image, request, job_id)

    return {"message": "Job queued", "job_id": job_id}


def process_image(request: ImageGenRequest, job_id: str):
    try:
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

            payload["init_image"] = request.init_image
            payload["denoising_strength"] = request.denoising_strength or 0.75  # Fallback
            endpoint = f"{request.api_url}/sdapi/v1/img2img"
        else:
            raise ValueError("Invalid mode")

        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            result = response.json()
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["result"] = result
        else:
            raise ValueError(f"Image generation failed: {response.text}")

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)


@router.get("/generate/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return job_status[job_id]
