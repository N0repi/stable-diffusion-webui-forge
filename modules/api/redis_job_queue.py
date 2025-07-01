# redis_job_queue.py

import json
import uuid
import time
import os
from typing import Dict, Optional, Any
import redis.asyncio as redis
from fastapi import HTTPException

class RedisJobQueue:
    def __init__(self):
        # Use the same Redis connection pattern as the existing redis-subscriber
        self.redis_url = os.getenv("UPSTASH_REDIS_REDIS_URL")
        self.redis_token = os.getenv("UPSTASH_REDIS_REDIS_TOKEN")
        
        if not self.redis_url or not self.redis_token:
            raise ValueError("Missing Redis environment variables")
        
        # Initialize Redis connection (matching existing redis-subscriber pattern)
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        
        # Queue and job keys
        self.job_queue_key = "forge:job_queue"
        self.job_status_prefix = "forge:job_status:"
        self.job_result_prefix = "forge:job_result:"
        
        # Job TTL (24 hours)
        self.job_ttl = 86400
        
    async def create_job(self, request_data: Dict[str, Any]) -> str:
        """Create a new job and add it to the Redis queue"""
        job_id = str(uuid.uuid4())
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "request": request_data,
            "created_at": time.time(),
            "status": "queued"
        }
        
        # Store job data in Redis
        await self.redis.setex(
            f"{self.job_status_prefix}{job_id}",
            self.job_ttl,
            json.dumps(job_data)
        )
        
        # Add job to queue (FIFO)
        await self.redis.lpush(self.job_queue_key, job_id)
        
        print(f"🧵 Job {job_id} created and queued in Redis")
        return job_id
    
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
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and result"""
        # Get job data
        job_data_str = await self.redis.get(f"{self.job_status_prefix}{job_id}")
        if not job_data_str:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job_data = json.loads(job_data_str)
        
        # If job is completed, get the result
        if job_data.get("status") == "completed" and "result" in job_data:
            result_str = await self.redis.get(f"{self.job_result_prefix}{job_id}")
            if result_str:
                job_data["result"] = json.loads(result_str)
        
        return job_data
    
    async def cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        # This could be implemented to remove old job data
        # For now, we rely on TTL
        pass
    
    async def close(self):
        """Close Redis connection"""
        await self.redis.close()

# Global instance
job_queue = RedisJobQueue() 