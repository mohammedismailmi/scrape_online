# main.py
import asyncio
from browser_use import Agent
from ollama_llm import OllamaLLM

async def main():
    agent = Agent(
        task="Navigate to https://github.com/trending/python?since=weekly and list the top trending Python projects on GitHub this week.",
        llm=OllamaLLM(),
        headless=False  # Show browser window
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
