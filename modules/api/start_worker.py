#!/usr/bin/env python3
# start_worker.py

import asyncio
import os
import sys
import signal
from redis_worker import start_worker

async def main():
    """Start the Redis job worker"""
    print("🚀 Starting Redis job worker...")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down worker...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await start_worker()
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped by user")
    except Exception as e:
        print(f"❌ Worker error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
