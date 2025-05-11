from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import os
import uuid
import time
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import json

from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction
from langchain_google_genai import ChatGoogleGenerativeAI


# MCP protocol models
class ContentBlock(BaseModel):
    type: str
    content: Union[str, Dict[str, Any]]


class MCPMessage(BaseModel):
    role: str
    content: List[ContentBlock]


class MCPRequest(BaseModel):
    messages: List[MCPMessage]
    stream: bool = False
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1024


class MCPResponse(BaseModel):
    id: str
    model: str
    created: int
    message: MCPMessage
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


# Simple shell command tool
class ShellTool(BaseTool):
    name = "shell"
    description = "Run shell commands on the system"
    
    def _run(self, cmd: str) -> str:
        """Run a shell command and return the output."""
        import subprocess
        try:
            return subprocess.check_output(cmd, shell=True, text=True)
        except subprocess.CalledProcessError as e:
            return f"Error executing command: {e}"


# Browser tool for web browsing
class BrowserTool(BaseTool):
    name = "browser"
    description = "Browse a URL and get the content of a webpage"
    
    def _run(self, url: str) -> str:
        """
        Fetch and process content from a URL.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            Processed content from the webpage
        """
        try:
            # Set a user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Process text: break into lines and remove leading/trailing whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit the length of the returned text to avoid overwhelming the model
            max_length = 4000
            if len(text) > max_length:
                text = text[:max_length] + "... [content truncated]"
            
            return text
        except Exception as e:
            return f"Error browsing URL: {e}"


# MCP Agent using LangChain and Gemini
class SimpleMCPAgent:
    """
    A simplified MCP Agent using LangChain and Gemini.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the simple MCP agent.
        
        Args:
            api_key: The Google API key (defaults to GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please provide it or set GOOGLE_API_KEY environment variable.")
        
        # Set up the LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            top_p=1.0,
            convert_system_message_to_human=True
        )
        
        # Set up tools
        self.tools = [ShellTool(), BrowserTool()]
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Set up the agent prompt
        self.prompt_template = """
        You are an AI assistant that helps users by answering questions and executing commands.
        
        USER QUERY: {input}
        
        Available tools:
        {tools}
        
        When you need to use a tool, format your response as follows:
        ```tool
        tool_name: parameter
        ```
        
        For example:
        ```tool
        shell: ls -la
        ```
        
        Or:
        ```tool
        browser: https://www.example.com
        ```
        
        Wait for tool output before continuing. Do not simulate or predict tool output.
        Only use tools when necessary. If no tool is needed, just provide your response directly.
        
        If you previously received tool output, consider it in your response.
        
        {previous_steps}
        
        Think step by step to respond effectively.
        """
        
        # Create the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["input", "tools", "previous_steps"]
        )
    
    def _format_prompt(self, messages: List[MCPMessage]) -> str:
        """Format MCP messages into a text prompt for the LLM."""
        formatted_messages = []
        
        for message in messages:
            if message.role == "user":
                text_content = "".join([
                    block.content if isinstance(block.content, str) else str(block.content)
                    for block in message.content if block.type == "text"
                ])
                formatted_messages.append(f"USER: {text_content}")
            else:
                text_content = "".join([
                    block.content if isinstance(block.content, str) else str(block.content)
                    for block in message.content if block.type == "text"
                ])
                formatted_messages.append(f"ASSISTANT: {text_content}")
        
        return "\n".join(formatted_messages)
    
    def parse_tool_call(self, text: str) -> Optional[tuple]:
        """
        Parse a tool call from the assistant's output.
        
        Args:
            text: The raw response from the LLM
            
        Returns:
            A tuple of (tool_name, tool_param) if a tool call is found, otherwise None
        """
        # Find any tool call blocks in the response
        tool_pattern = r'```tool\s*\n([a-zA-Z_]+):\s*(.*?)```'
        match = re.search(tool_pattern, text, re.DOTALL)
        
        if match:
            tool_name = match.group(1).strip()
            tool_param = match.group(2).strip()
            return (tool_name, tool_param)
        
        return None
    
    def execute_tool(self, tool_name: str, tool_param: str) -> str:
        """
        Execute a tool with the given parameter.
        
        Args:
            tool_name: The name of the tool to execute
            tool_param: The parameter to pass to the tool
            
        Returns:
            The output from the tool
        """
        if tool_name not in self.tool_map:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_map.keys())}"
        
        tool = self.tool_map[tool_name]
        try:
            return tool._run(tool_param)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def process_response(self, response_text: str) -> str:
        """
        Clean and format the final response.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            The processed response
        """
        # Remove any remaining tool blocks
        cleaned_text = re.sub(r'```tool\s*\n.*?```', '', response_text, flags=re.DOTALL)
        
        # Remove any "thinking out loud" or step-by-step reasoning
        # This is optional - comment out if you want to keep the reasoning
        thinking_patterns = [
            r'Step \d+:.*?\n',
            r'Let me .*?\n',
            r'I will .*?\n',
            r'First,.*?\n',
            r'Next,.*?\n',
            r'Finally,.*?\n',
        ]
        
        for pattern in thinking_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request and return a response."""
        # Format the messages into a text prompt
        input_text = self._format_prompt(request.messages)
        
        # Get tools descriptions
        tools_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        # Conversation state
        previous_steps = ""
        max_iterations = 5  # Prevent infinite loops
        current_iteration = 0
        
        # Main agent loop
        while current_iteration < max_iterations:
            current_iteration += 1
            
            # Create LLM chain for this iteration
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
            
            # Generate assistant response
            response_text = await llm_chain.arun(
                input=input_text,
                tools=tools_str,
                previous_steps=previous_steps
            )
            
            # Check if the response contains a tool call
            tool_call = self.parse_tool_call(response_text)
            
            if tool_call:
                # Extract tool details
                tool_name, tool_param = tool_call
                
                # Log the tool call
                tool_log = f"Tool Call: {tool_name}({tool_param})\n"
                
                # Execute the tool
                tool_output = self.execute_tool(tool_name, tool_param)
                
                # Add to conversation state
                previous_steps += f"\nAssistant: {response_text}\n\nTool Output: {tool_output}\n"
                
                # Continue to next iteration
                continue
            
            # No tool call found, this is the final response
            break
        
        # Process the final response
        final_response = self.process_response(response_text)
        
        # Create and return the MCP response
        return MCPResponse(
            id=f"mcp-{uuid.uuid4().hex[:8]}",
            model=request.model,
            created=int(time.time()),
            message=MCPMessage(
                role="assistant",
                content=[ContentBlock(type="text", content=final_response)]
            )
        )


# Create a singleton agent instance
simple_mcp_agent = SimpleMCPAgent(os.environ.get("GOOGLE_API_KEY")) 