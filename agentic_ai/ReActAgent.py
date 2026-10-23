#Third party tools
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
#App imports
from agentic_ai.common import get_llm_model
g_llm_model = get_llm_model()


PROMPT = """
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
"""

@tool
def summarize_text(text: str)->str:
    """

    :param text:
    :return: summary of the text passed as input
    """
    input_text = f"Summarise this text in less than 200 words:{text}"
    summary = g_llm_model.invoke(input_text)
    return summary

@tool
def basic_math_operations(a:float, b:float, operation:str)->float:
    """

    :param a:
    :param b:
    :param operation:
    :return: output of the math operation
    """
    match operation:
        case '+':
            return a + b
        case '-':
            return a - b
        case '*':
            return a * b
        case '/':
            return a/b if b!=0 else 0
        case _:
            raise ValueError("Unexpected char in input")

# LangChain agent
agent = create_agent(
    model=g_llm_model,
    tools=[summarize_text, basic_math_operations],
    system_prompt=PROMPT
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Multiply 12 by 5, then add 10."}
    ]
})


print(result["messages"][-1].content)
