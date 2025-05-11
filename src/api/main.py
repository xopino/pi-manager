from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pi Manager", description="A FastAPI application for managing Raspberry Pi with Llama MCP agent")

@app.get("/")
def read_root():
    return {
        "name": "Pi Manager with Llama MCP Agent",
        "version": "0.1.0",
        "endpoints": {
            "mcp_chat": "/mcp/chat",
            "mcp_stream": "/mcp/chat/stream",
            "mcp_websocket": "/mcp/chat/ws",
            "mcp_models": "/mcp/models"
        }
    }
