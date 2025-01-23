import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Final

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from surfkit.server.routes import task_router

from .agent import Agent

# Configure logging
logger: Final = logging.getLogger("robbieg2")
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Ensure logs are flushed immediately
handler.flush = sys.stdout.flush
logger.addHandler(handler)
logger.propagate = False

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
ALLOW_METHODS = os.getenv("ALLOW_METHODS", "*").split(",")
ALLOW_HEADERS = os.getenv("ALLOW_HEADERS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the agent type before the server comes live
    Agent.init()
    yield


app = FastAPI(lifespan=lifespan)  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOW_METHODS,
    allow_headers=ALLOW_HEADERS,
)

app.include_router(task_router(Agent))

if __name__ == "__main__":
    port = os.getenv("SERVER_PORT", "9090")
    reload = os.getenv("SERVER_RELOAD", "true") == "true"
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    uvicorn.run(
        "robbieg2.server:app",
        host=host,
        port=int(port),
        reload=reload,
        reload_excludes=[".data"],
        log_config=None,  # Disable default Uvicorn log configuration
    )
