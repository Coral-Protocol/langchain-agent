from typing import Sequence, override, Literal
from langchain_core.messages.tool import tool_call
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain.memory import SimpleMemory
from langchain_mcp_adapters.client import BaseTool, MultiServerMCPClient
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

import requests
import asyncio
from os import getenv
import sys
from dotenv import load_dotenv
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(rich_tracebacks=True))


def asserted_env(name: str, extra_msg: str = "") -> str:
    value = getenv(name, None)
    if value is None:
        logger.error(f"Option '{name}' not provided! {extra_msg}")
        sys.exit(1)
    return value


class InMemoryHistory(  # pyright: ignore[reportUnsafeMultipleInheritance]
    BaseChatMessageHistory, BaseModel
):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    @override
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    @override
    def clear(self) -> None:
        self.messages = []


class ToolRunner:
    tool_by_name: dict[str, BaseTool]

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self.tool_by_name = {}
        for tool in tools:
            self.tool_by_name[tool.name] = tool

    async def run_tool(self, call: ToolCall) -> ToolMessage:
        if call["name"] == "":
            return ToolMessage(
                content=f"Failed to run tool - no tool name passed!",
                tool_call_id=call["id"],
            )
        tool = self.tool_by_name.get(call["name"], None)
        if tool is None:
            return ToolMessage(
                content=f"Failed to run tool - tool '{call["name"]}' not found",
                tool_call_id=call["id"],
            )
        try:
            tool_result = await tool.arun(call["args"])
            message = ToolMessage(content=tool_result, tool_call_id=call["id"])
            logger.debug("tool_result: %s", tool_result)
        except Exception as e:
            content = f"Error running tool '{call["name"]}' - {e}"
            logger.warning(content)
            return ToolMessage(content=content)
        return message


def claim(amount: int, claim_type: Literal["coral", "usd"] = "coral"):
    CORAL_RUNTIME = getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if CORAL_RUNTIME is None:
        logger.warning("Not orchestrated - skipping Coral Server claim")
        return True

    coral_api_url = asserted_env(
        "CORAL_API_URL",
        "This is set by Coral Server in orchestration, and CORAL_ORCHESTRATION_RUNTIME is set - are you running an older version of Coral Server?",
    )
    coral_session_id = asserted_env(
        "CORAL_SESSION_ID",
        "This is set by Coral Server in orchestration, and CORAL_ORCHESTRATION_RUNTIME is set - are you running an older version of Coral Server?",
    )
    try:
        response = requests.post(
            f"{coral_api_url}/api/v1/internal/claim/{coral_session_id}",
            headers={"Content-Type": "application/json"},
            json={"amount": {"type": claim_type, "amount": amount}},
        )

        if response.status_code == 200:
            logger.info(f"Claimed {amount} {claim_type}")
            return True
        else:
            logger.error(
                f"Failed to claim {amount} {claim_type} - got {response.status_code} status"
            )
            return False

    except:
        logger.exception(f"Error claiming {amount} {claim_type}")
        return False


def make_history():
    store = InMemoryHistory()

    def get_history() -> BaseChatMessageHistory:
        return store

    return get_history


async def main():
    # What orchestration runtime we are running in (docker, executable, etc.). None if we aren't orchestrated (i.e. devmode)
    CORAL_RUNTIME = getenv("CORAL_ORCHESTRATION_RUNTIME", None)

    CORAL_CONNECTION_URL = asserted_env("CORAL_CONNECTION_URL")

    MODEL_NAME = asserted_env("MODEL_NAME")
    MODEL_PROVIDER = asserted_env("MODEL_PROVIDER")
    MODEL_API_KEY = asserted_env("MODEL_API_KEY")
    MODEL_BASE_URL = getenv("MODEL_BASE_URL")

    # Load .env file if we are in dev mode for convenience
    if CORAL_RUNTIME is None:
        _ = load_dotenv()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("coral_instruction"),
            SystemMessage(
                content="You are an agent. Please create a thread, and send a message into it."
            ),
            MessagesPlaceholder("coral_messages", optional=True),
            MessagesPlaceholder("history"),
        ]
    )

    get_history = make_history()

    model = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
    )

    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_CONNECTION_URL,
                "timeout": 300000,
                "sse_read_timeout": 300000,
            }
        }
    )
    logger.info(f"Connecting to Coral @ '{CORAL_CONNECTION_URL}'")
    try:
        tools = await client.get_tools()
    except:
        logger.exception("Failed to get MCP tools")
        sys.exit(1)

    chain = prompt | model.bind_tools(tools)
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        # input_messages_key="history",
        history_messages_key="history",
    )

    tool_runner = ToolRunner(tools)

    async with client.session("coral") as coral_session:
        coral_instruction = (
            await load_mcp_resources(coral_session, uris="coral://Instruction.resource")
        )[0]

        for _ in range(10):
            coral_messages = (
                await load_mcp_resources(coral_session, uris="coral://Message.resource")
            )[0]
            logger.info("coral_messages: %s", coral_messages)

            # history = get_history()
            # logger.debug("pre call history => \n%s", history.messages)

            step_result: BaseMessage = await chain_with_history.ainvoke(
                {  # We pass in our loaded resources here (as_string is safe here because we know these resources always return text)
                    "coral_instruction": [
                        SystemMessage(content=coral_instruction.as_string())
                    ],
                    "coral_messages": [
                        SystemMessage(content=coral_messages.as_string())
                    ],
                }
            )

            history = get_history()
            logger.debug("post call history => \n%s", history.messages)

            if type(step_result) is AIMessage:
                tool_calls = step_result.tool_calls
                for call in tool_calls:
                    history.add_message(await tool_runner.run_tool(call))

            # history.append(step_result["output"])
            logger.debug("step_result: %s", step_result)


if __name__ == "__main__":
    asyncio.run(main())
