from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from .simple_agent_routes import router as simple_mcp_router

app = FastAPI(title="Pi Manager", description="A FastAPI application for managing Raspberry Pi with Llama MCP agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the simple MCP router
app.include_router(simple_mcp_router)

# Get the directory of this file
current_dir = Path(__file__).parent.parent
static_dir = current_dir / "static"

# Mount the static files directory
app.mount("/chat", StaticFiles(directory=str(static_dir), html=True), name="chat")

@app.get("/")
def read_root():
    return {
        "name": "Pi Manager with Llama MCP Agent",
        "version": "0.1.0",
        "endpoints": {
            "simple_mcp_chat": "/simple-mcp/chat",
            "simple_mcp_status": "/simple-mcp/status",
            "chat_interface": "/chat"
        }
    }
