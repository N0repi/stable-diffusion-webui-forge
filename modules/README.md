# Forge Container with Redis Job Queue

This container implements a Redis-based job queue system to avoid Cloudflare 524 timeouts during long image generation jobs.

## Architecture

### Components

1. **FastAPI Server** (`api/api.py`) - Main API server
2. **Job Queue Router** (`api/job_queue_router.py`) - Handles job creation and status polling
3. **Redis Job Queue** (`api/redis_job_queue.py`) - Redis-based job storage and management
4. **Redis Worker** (`api/redis_worker.py`) - Background worker that processes jobs
5. **Supervisor** - Manages both FastAPI server and Redis worker processes

### Workflow

1. **Job Creation**: Client calls `/queue/generate` → Job is immediately queued in Redis → Returns `job_id`
2. **Job Processing**: Redis worker continuously polls for jobs and processes them in background
3. **Status Polling**: Client polls `/queue/generate/status/{job_id}` → Returns current job status

### Benefits

- ✅ **No Cloudflare 524 timeouts** - Jobs are queued immediately, no long-held requests
- ✅ **Scalable** - Multiple workers can process jobs from the same queue
- ✅ **Persistent** - Jobs survive container restarts
- ✅ **Fault-tolerant** - Failed jobs are marked with error details

## Environment Variables

Required Redis environment variables (same as existing redis-subscriber):

```bash
UPSTASH_REDIS_REDIS_URL=your_redis_url
UPSTASH_REDIS_REDIS_TOKEN=your_redis_token
```

## Redis Keys

- `forge:job_queue` - FIFO queue of job IDs
- `forge:job_status:{job_id}` - Job metadata and status
- `forge:job_result:{job_id}` - Job results (stored separately for large data)

## Running

The container uses Supervisor to run both processes:

```bash
# Build and run
docker build -t forge-container .
docker run -p 8000:8000 forge-container
```

## Monitoring

Check logs for both processes:

```bash
# FastAPI server logs
tail -f /var/log/supervisor/fastapi_stdout.log

# Redis worker logs
tail -f /var/log/supervisor/redis_worker_stdout.log
```

## Migration from In-Memory Queue

The old in-memory job queue has been replaced with Redis-based storage. This eliminates the Cloudflare 524 timeout issue by:

1. Immediately returning a `job_id` when a job is submitted
2. Processing jobs asynchronously in the background
3. Storing job status and results in Redis for reliable polling
