## Standard library imports
from typing import Literal

# Third party imports
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

#App imports
from agentic_ai.common import get_llm_model
g_llm_model = get_llm_model()

PROMPT = PromptTemplate.from_template("""
You are a helpful assistant.

Available tools:
{tools}

Tool names: {tool_names}

Instructions:
- Use summarize_text when the user requests a summary.
- Use calculator for arithmetic.
- Use both tools when needed.
- Answer directly when no tool is required.
- If no tool was used, start with:
  "I haven't used any tool for this."

Use this format:

Question: {input}
Thought: consider which action is appropriate
Action: one of [{tool_names}]
Action Input: the input to the tool
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have enough information
Final Answer: your response

{agent_scratchpad}
""")
# Tool 1: Summarize text
@tool
def summarize_text(text: str) -> str:
    """Summarize a paragraph into a few concise sentences."""
    response = g_llm_model.invoke(
        f"Summarize this paragraph concisely:\n\n{text}"
    )
    return response.content


# Tool 2: Calculator
@tool
def calculator(
    a: float,
    b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> float:
    """Calculate using two numbers and an operation."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError("Unsupported operation")


# LangChain agent
agent = create_react_agent(
    llm=g_llm_model,
    tools=[summarize_text, calculator],
    prompt=PROMPT
)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[summarize_text, calculator],
    verbose=True
)

# Test
result = agent_executor.invoke({
    "input": "What is 25 multiplied by 16?"
})

print(result["output"])