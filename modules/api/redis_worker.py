# redis_worker.py

import asyncio
import json
import time
import uuid
import requests
import os
from typing import Dict, Any, Optional
import redis.asyncio as redis

class RedisJobWorker:
    def __init__(self):
        # Use the same Redis connection pattern as the existing redis-subscriber
        self.redis_url = os.getenv("UPSTASH_REDIS_REDIS_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_REDIS_TOKEN")
        
        if not self.redis_url or not self.redis_token:
            raise ValueError("Missing Redis environment variables")
        
        # Initialize Redis connection (matching existing redis-subscriber pattern)
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        
        # Queue and job keys (same as RedisJobQueue)
        self.job_queue_key = "forge:job_queue"
        self.job_status_prefix = "forge:job_status:"
        self.job_result_prefix = "forge:job_result:"
        
        # Job TTL (24 hours)
        self.job_ttl = 86400
        
        # Worker state
        self.running = False
        
    async def start(self):
        """Start the worker loop"""
        self.running = True
        print("🔄 Redis job worker started")
        
        while self.running:
            try:
                # Get next job from queue
                job_data = await self.get_next_job()
                if job_data:
                    await self.process_job(job_data)
                else:
                    # No jobs available, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ Worker error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def stop(self):
        """Stop the worker"""
        self.running = False
        await self.redis.close()
        
    async def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next job from the queue (FIFO)"""
        # Pop job from queue (blocking for 1 second)
        result = await self.redis.brpop(self.job_queue_key, timeout=1)
        if not result:
            return None
            
        job_id = result[1]
        
        # Get job data
        job_data_str = await self.redis.get(f"{self.job_status_prefix}{job_id}")
        if not job_data_str:
            return None
            
        job_data = json.loads(job_data_str)
        job_data["status"] = "processing"
        
        # Update status
        await self.redis.setex(
            f"{self.job_status_prefix}{job_id}",
            self.job_ttl,
            json.dumps(job_data)
        )
        
        return job_data
    
    async def update_job_status(self, job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None):
        """Update job status and store result/error"""
        # Get current job data
        job_data_str = await self.redis.get(f"{self.job_status_prefix}{job_id}")
        if not job_data_str:
            return
            
        job_data = json.loads(job_data_str)
        job_data["status"] = status
        job_data["updated_at"] = time.time()
        
        if result:
            job_data["result"] = result
            # Store result separately for larger data
            await self.redis.setex(
                f"{self.job_result_prefix}{job_id}",
                self.job_ttl,
                json.dumps(result)
            )
            
        if error:
            job_data["error"] = error
            
        # Update job status
        await self.redis.setex(
            f"{self.job_status_prefix}{job_id}",
            self.job_ttl,
            json.dumps(job_data)
        )
        
        print(f"📊 Job {job_id} status updated to: {status}")
    
    async def process_job(self, job_data: Dict[str, Any]):
        """Process a single job"""
        job_id = job_data["job_id"]
        request = job_data["request"]
        
        try:
            print(f"🟢 Starting process_job for job_id={job_id}")
            t0 = time.time()
            
            # Update status to processing
            await self.update_job_status(job_id, "processing")
            
            # Prepare payload
            payload = {
                "prompt": request["prompt"],
                "steps": 15,
                "seed": request.get("seed") or uuid.uuid4().int % (2**32),
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
            
            # Determine endpoint and add img2img specific data
            if request["mode"] == "txt2img":
                endpoint = f"{request['api_url']}/sdapi/v1/txt2img"
            elif request["mode"] == "img2img":
                if not request.get("init_image"):
                    raise ValueError("init_image is required for img2img mode")
                payload["init_images"] = [request["init_image"]]
                payload["denoising_strength"] = request.get("denoising_strength", 0.85)
                endpoint = f"{request['api_url']}/sdapi/v1/img2img"
            else:
                raise ValueError("Invalid mode")
            
            print(f"📤 Sending generation request to: {endpoint}")
            MAX_RETRIES = 3
            RETRY_DELAY = 3
            
            # Process with retries
            for attempt in range(MAX_RETRIES):
                post_start = time.time()
                try:
                    response = requests.post(endpoint, json=payload, timeout=420)
                except Exception as post_err:
                    print(f"⚠️ Request attempt {attempt + 1} failed: {post_err}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    else:
                        raise
                
                post_duration = time.time() - post_start
                print(f"📨 Attempt {attempt + 1}: POST completed in {post_duration:.2f}s")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"✅ JSON parsed successfully on attempt {attempt + 1}")
                        await self.update_job_status(job_id, "completed", result=result)
                        total = time.time() - t0
                        print(f"🏁 Job {job_id} completed successfully in {total:.2f}s")
                        return
                    except Exception as json_err:
                        print(f"❌ Invalid JSON on attempt {attempt + 1}: {json_err}")
                        if attempt < MAX_RETRIES - 1:
                            print("🔁 Retrying due to invalid JSON...")
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        else:
                            raise ValueError("Forge returned non-JSON response.")
                elif response.status_code == 404:
                    print(f"⚠️ Forge not ready (404) on attempt {attempt + 1}. Retrying...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"❌ Forge returned status {response.status_code} on attempt {attempt + 1}")
                    print(f"↩ Response text: {response.text[:300]}...")
                    raise ValueError(f"Image generation failed: {response.text}")
            
            # If we reach here, all attempts failed
            raise RuntimeError("All retry attempts failed for Forge inference")
            
        except Exception as e:
            print(f"❌ process_job failed for job_id {job_id}: {e}")
            await self.update_job_status(job_id, "failed", error=str(e))

# Worker instance
worker = RedisJobWorker()

# Start function for running the worker
async def start_worker():
    """Start the Redis job worker"""
    await worker.start()

if __name__ == "__main__":
    # Run the worker
    asyncio.run(start_worker()) 