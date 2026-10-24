"""
Example usage of the Conversational Agent

This script demonstrates:
1. Building the agent
2. Having a multi-turn conversation
3. Testing memory recall
"""

import asyncio
import logging
from agentic_ai.conversational_Agent.main import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_conversation():
    """Example conversation demonstrating memory capabilities."""

    print("\n" + "="*60)
    print("Conversational Agent - Example Usage")
    print("="*60 + "\n")

    # Build the agent
    print("Building agent...")
    agent = await build_agent()
    print("Agent ready!\n")

    user_id = "example_user"
    session_id = None

    # Turn 1: Introduction
    print("Turn 1: User introduces themselves")
    response, session_id = await agent.chat(
        user_message="Hi! My name is Alice and I love playing guitar.",
        user_id=user_id,
        session_id=session_id
    )
    print(f"User: Hi! My name is Alice and I love playing guitar.")
    print(f"Agent: {response}\n")

    # Turn 2: Ask about a topic
    print("Turn 2: User asks about learning")
    response, session_id = await agent.chat(
        user_message="I want to learn jazz guitar. Any tips?",
        user_id=user_id,
        session_id=session_id
    )
    print(f"User: I want to learn jazz guitar. Any tips?")
    print(f"Agent: {response}\n")

    # Turn 3: Change topic
    print("Turn 3: User changes topic")
    response, session_id = await agent.chat(
        user_message="What's the weather like today?",
        user_id=user_id,
        session_id=session_id
    )
    print(f"User: What's the weather like today?")
    print(f"Agent: {response}\n")

    # Turn 4: Test memory recall
    print("Turn 4: Test memory recall")
    response, session_id = await agent.chat(
        user_message="What's my name and what do I like?",
        user_id=user_id,
        session_id=session_id
    )
    print(f"User: What's my name and what do I like?")
    print(f"Agent: {response}\n")

    # Turn 5: Test semantic memory
    print("Turn 5: Test semantic memory (related to earlier topic)")
    response, session_id = await agent.chat(
        user_message="Can you remind me what musical style I wanted to learn?",
        user_id=user_id,
        session_id=session_id
    )
    print(f"User: Can you remind me what musical style I wanted to learn?")
    print(f"Agent: {response}\n")

    print("="*60)
    print(f"Session ID: {session_id}")
    print("Example complete!")
    print("="*60)


async def example_multi_session():
    """Example showing memory across multiple sessions."""

    print("\n" + "="*60)
    print("Multi-Session Memory Example")
    print("="*60 + "\n")

    agent = await build_agent()
    user_id = "multi_session_user"

    # Session 1
    print("SESSION 1:")
    response, session1 = await agent.chat(
        user_message="My favorite color is blue.",
        user_id=user_id,
        session_id=None
    )
    print(f"User: My favorite color is blue.")
    print(f"Agent: {response}\n")

    # Session 2 (new session, but should remember via Neo4j)
    print("SESSION 2 (New session):")
    response, session2 = await agent.chat(
        user_message="What was my favorite color?",
        user_id=user_id,
        session_id=None  # New session
    )
    print(f"User: What was my favorite color?")
    print(f"Agent: {response}\n")

    print(f"Session 1 ID: {session1}")
    print(f"Session 2 ID: {session2}")
    print("Note: Sessions are different, but memory is retained via Neo4j!")
    print("="*60)


if __name__ == "__main__":
    print("\nRunning example conversation...")
    asyncio.run(example_conversation())

    print("\n\nPress Enter to run multi-session example...")
    input()

    asyncio.run(example_multi_session())