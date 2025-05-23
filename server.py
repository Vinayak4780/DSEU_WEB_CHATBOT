from pydantic import BaseModel
import multiprocessing as mp

# 1) Ensure child processes use 'spawn' for CUDA-safe multiprocessing
mp.set_start_method("spawn", force=True)

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from main import handle_query

app = FastAPI()

# 2) Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3) Prepare your pools at module load
#    - ProcessPool for real parallelism (zero-shot, model inference)
#    - ThreadPool for any lightweight I/O tasks
ctx          = mp.get_context("spawn")
process_pool = ProcessPoolExecutor(max_workers=4, mp_context=ctx)
thread_pool  = ThreadPoolExecutor(max_workers=20)

class Create(BaseModel):
    prompt: str

@app.post("/prompt")
async def create(post: Create):
    """
    Offload the CPU/GPU-heavy handle_query into the process pool,
    so the FastAPI event loop stays fully free to accept new requests.
    """
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        process_pool,        # CPU-bound pool
        handle_query,        # function to run
        post.prompt          # its argument
    )
    return {"response": response}
@app.get("/")
async def start():
    return {"message": "Initiating ChatBot....."}

