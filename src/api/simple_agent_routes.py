from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from ..agent.simple_mcp_agent import simple_mcp_agent, MCPRequest, MCPResponse

# Create a router for simple MCP API endpoints
router = APIRouter(prefix="/simple-mcp", tags=["Simple MCP"])


# Dependency to check if API key is configured
def verify_api_key():
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Google API key not configured. Set the GOOGLE_API_KEY environment variable."
        )
    return True


@router.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: MCPRequest) -> MCPResponse:
    """Endpoint for simple MCP chat completion."""
    try:
        return await simple_mcp_agent.process_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Get the status of the simple MCP agent."""
    return {
        "status": "online",
        "agent": "SimpleMCPAgent",
        "model": "gemini-1.5",
        "tools": ["shell"]
    } 