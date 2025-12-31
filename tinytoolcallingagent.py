import asyncio
import json
import sys
import os
import argparse
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any

from huggingface_hub import InferenceClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SYSTEM_PROMPT = """You are a helpful assistant that can solve any task."""

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, message):
        self.choices = [MockChoice(message)]

class MockInferenceClient:
    def chat_completion(self, messages, max_tokens=1000, tools=None):
        last_msg = messages[-1]
        content = last_msg.get("content", "").lower()
        role = last_msg.get("role", "")

        # Simple heuristic to drive the AP2 demo flow
        if role == "user":
            if "search" in content:
                # Trigger search
                return MockResponse(MockMessage(
                    content="I will search for a Smartwatch.",
                    tool_calls=[
                        type('obj', (object,), {
                            "function": type('obj', (object,), {
                                "name": "search_products",
                                "arguments": json.dumps({"query": "Smartwatch"})
                            })
                        })
                    ]
                ))
            elif "negotiate" in content:
                # Trigger negotiate
                return MockResponse(MockMessage(
                    content="I will negotiate the price for the QuantumX Smartwatch.",
                    tool_calls=[
                        type('obj', (object,), {
                            "function": type('obj', (object,), {
                                "name": "negotiate_price",
                                "arguments": json.dumps({"product_id": "prod_001", "offer_price": 260.0})
                            })
                        })
                    ]
                ))
            elif "checkout" in content:
                # Trigger checkout
                return MockResponse(MockMessage(
                    content="I will proceed to checkout.",
                    tool_calls=[
                        type('obj', (object,), {
                            "function": type('obj', (object,), {
                                "name": "checkout",
                                "arguments": json.dumps({"product_id": "prod_001", "confirmed_price": 260.0})
                            })
                        })
                    ]
                ))
            elif "what product" in content:
                return MockResponse(MockMessage(
                    content="You just purchased the QuantumX Smartwatch for $260.00 using Google Pay via the AP2 Agentic Checkout protocol."
                ))
            else:
                return MockResponse(MockMessage(content="I am a mock agent. I didn't understand that."))

        return MockResponse(MockMessage(content="Proceeding..."))

class TinyToolCallingAgent:
    def __init__(self, memory_file: Optional[str] = None, mock: bool = False):
        # Initialize MCP client session and (LLM) model
        self.mcp_client_session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        if mock:
            print("Using MockInferenceClient...")
            self.model = MockInferenceClient()
        else:
            self.model = InferenceClient(model="Qwen/Qwen2.5-Coder-32B-Instruct", provider="hf-inference")

        self.system_prompt = SYSTEM_PROMPT
        self.tools = []
        self.memory_file = memory_file
        self.messages: List[Dict[str, str]] = []

        self._load_memory()

    def _load_memory(self):
        """Load conversation history from memory file if it exists."""
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.messages = json.load(f)
                print(f"Loaded {len(self.messages)} messages from memory.")
            except Exception as e:
                print(f"Error loading memory: {e}")

        # Ensure system prompt is present if memory was empty or didn't have it
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages = [{"role": "system", "content": self.system_prompt}] + self.messages

    def _save_memory(self):
        """Save conversation history to memory file."""
        if self.memory_file:
            try:
                with open(self.memory_file, 'w') as f:
                    json.dump(self.messages, f, indent=2)
            except Exception as e:
                print(f"Error saving memory: {e}")

    async def _call_model(self, messages, tools=None):
        """Call the model with the given messages and tools.

        Args:
            messages: List of messages to send to the model.
            tools: List of tools available for the model to use.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.model.chat_completion(messages, max_tokens=1000, tools=tools)
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not server_script_path.endswith(".py") and not server_script_path.endswith(".js"):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if server_script_path.endswith(".py") else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.mcp_client_session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.mcp_client_session.initialize()

        # List available tools
        response = await self.mcp_client_session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def process_query(self, query: str) -> str:
        """Process a query using model and available tools"""
        # Append user query to memory
        self.messages.append({"role": "user", "content": query})
        self._save_memory()

        # List available tools for the model call
        tools = [
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
            }
            for tool in self.tools
        ]

        # Initial model call with tools
        # We pass self.messages which includes history
        response = await self._call_model(self.messages, tools=tools)
        # print("Initial response:", response)

        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message

        # Add assistant's initial response (thought/content) to memory
        if message.content:
            self.messages.append({"role": "assistant", "content": message.content})
            final_text.append(message.content)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool call
                try:
                    result = await self.mcp_client_session.call_tool(tool_name, tool_args)
                    tool_output = result.content[0].text
                except Exception as e:
                    tool_output = str(e)

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                tool_msg = {"role": "user", "content": f"Tool '{tool_name}' output: {tool_output}"}
                self.messages.append(tool_msg)

            # Get next response from model after tool outputs
            # For the mock, we just want a final confirmation or pass
            response = await self._call_model(self.messages)
            if response.choices[0].message.content:
                final_content = response.choices[0].message.content
                # Avoid duplicates if mock returns same content
                if final_content != message.content:
                    self.messages.append({"role": "assistant", "content": final_content})
                    final_text.append(final_content)

        self._save_memory()
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!\nType your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {type(e).__name__}: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    parser = argparse.ArgumentParser(description="TinyToolCallingAgent with MCP")
    parser.add_argument("server_script", help="Path to the server script (.py or .js)")
    parser.add_argument("--memory-file", help="Path to JSON file for persistent memory", default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock LLM instead of Hugging Face")

    args = parser.parse_args()

    agent = TinyToolCallingAgent(memory_file=args.memory_file, mock=args.mock)
    try:
        await agent.connect_to_server(args.server_script)
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
