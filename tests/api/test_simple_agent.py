from fastapi.testclient import TestClient
import pytest
import os
from unittest.mock import patch, AsyncMock

from src.api.main import app
from src.agent.simple_mcp_agent import ContentBlock, MCPMessage, MCPRequest


client = TestClient(app)


@pytest.fixture
def mock_api_key():
    """Mock the API key for testing."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-api-key"}):
        yield


@pytest.fixture
def mock_agent_response():
    """Mock the agent's response."""
    with patch("src.agent.simple_mcp_agent.SimpleMCPAgent.process_request") as mock:
        # Create a fake response
        from src.agent.simple_mcp_agent import MCPResponse
        
        # Make it an AsyncMock to properly mock the async function
        mock_async = AsyncMock()
        mock_async.return_value = MCPResponse(
            id="mcp-test1234",
            model="gemini-1.5-pro",
            created=1234567890,
            message=MCPMessage(
                role="assistant",
                content=[ContentBlock(type="text", content="This is a test response")]
            )
        )
        mock.side_effect = mock_async
        yield mock


def test_status_endpoint():
    """Test that the status endpoint returns the expected response."""
    response = client.get("/simple-mcp/status")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["agent"] == "SimpleMCPAgent"


def test_chat_endpoint(mock_api_key, mock_agent_response):
    """Test that the chat endpoint processes requests correctly."""
    # Create a test request
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "content": "Hello, world!"}]
            }
        ],
        "stream": False,
        "model": "gemini-1.5-pro"
    }
    
    # Send the request to the API
    response = client.post("/simple-mcp/chat", json=request_data)
    
    # Verify the response
    assert response.status_code == 200
    assert mock_agent_response.called
    assert response.json()["message"]["content"][0]["content"] == "This is a test response" 