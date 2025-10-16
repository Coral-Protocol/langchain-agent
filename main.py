from typing import Sequence, override, Literal
from langchain_core.tools import tool
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import BaseTool, MultiServerMCPClient
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from pydantic import BaseModel, Field

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


class ClaimError(Exception):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class ClaimHandler:
    _remaining: float | None = None
    _currency: Literal["coral", "micro_coral", "usd"]

    def __init__(
        self, currency: Literal["coral", "micro_coral", "usd"] = "micro_coral"
    ) -> None:
        if currency not in ["coral", "micro_coral", "usd"]:
            raise ValueError("invalid currency %s" % self._currency)
        self._currency = currency

    def no_budget(self) -> bool:
        """Returns true if we *know* we have no budget remaining, false if we either have >0 remaining - or we don't know our budget yet (no claim calls yet)"""
        return (self._remaining is not None) and self._remaining <= 0

    def remaining(self) -> float | None:
        """
        Get the last known remaining budget amount, if known.

        Returns None if we have never called claim() yet - this does NOT necessarily mean we have no budget

        """
        return self._remaining

    def currency(self) -> Literal["coral", "micro_coral", "usd"]:
        """Returns the currency this claim handler uses ('coral', 'micro_coral' or 'usd')"""
        return self._currency

    def claim(self, amount: float) -> float:
        """
        Send a claim request to the Coral Server, returning the remaining budget.
        All units are in the currency this class was constructed with
        """
        CORAL_SEND_CLAIMS = getenv(
            "CORAL_SEND_CLAIMS", "0"
        )  # is set to 1 by coral server when running remotely
        if CORAL_SEND_CLAIMS == "0":
            logger.warning("Not orchestrated - skipping Coral Server claim")
            return True

        coral_api_url = asserted_env(
            "CORAL_API_URL",
            "This should be set by Coral Server in orchestration, and CORAL_SEND_CLAIMS is 1 - make sure you are not setting these manually!",
        )
        coral_session_id = asserted_env(
            "CORAL_SESSION_ID",
            "This should be set by Coral Server in orchestration, and CORAL_SEND_CLAIMS is 1 - make sure you are not setting these manually!",
        )
        try:
            response = requests.post(
                f"{coral_api_url}/api/v1/internal/claim/{coral_session_id}",
                headers={"Content-Type": "application/json"},
                json={"amount": {"type": self._currency, "amount": amount}},
            )

            if response.status_code == 200:
                budget = response.json()
                remaining = float(budget["remainingBudget"])  # in micro-coral
                match self._currency:
                    case "coral":
                        remaining = remaining * 1_000_000
                    case "micro_coral":
                        pass
                    case "usd":
                        remaining = (
                            remaining * 1_000_000 * float(budget["coralUsdPrice"])
                        )
                logger.info(
                    f"Claimed {amount} {self._currency} - remaining budget: {remaining} {self._currency}"
                )
                self._remaining = remaining
                return remaining
            else:
                raise ClaimError(
                    f"Failed to claim {amount} {self._currency} - got {response.status_code} status",
                    response,
                )

        except Exception as e:
            self._remaining = 0
            raise e


@tool()
def test_tool():
    """Test tool, should be called as often as possible"""
    global claim_handler
    if claim_handler.no_budget():
        raise RuntimeError("Out of budget")

    logger.info("WE ARE DOING A TOOL")
    _ = claim_handler.claim(900)

    return "Success!"


async def main():
    # What orchestration runtime we are running in (docker, executable, etc.). None if we aren't orchestrated (i.e. devmode)
    CORAL_RUNTIME = getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    # Load .env file if we are in dev mode for convenience
    if CORAL_RUNTIME is None:
        _ = load_dotenv()

    CORAL_CONNECTION_URL = asserted_env("CORAL_CONNECTION_URL")

    SYSTEM_PROMPT = asserted_env("SYSTEM_PROMPT")
    MODEL_NAME = asserted_env("MODEL_NAME")
    MODEL_PROVIDER = asserted_env("MODEL_PROVIDER")
    MODEL_API_KEY = asserted_env("MODEL_API_KEY")
    MODEL_BASE_URL = getenv("MODEL_BASE_URL")

    global claim_handler
    claim_handler = ClaimHandler("micro_coral")

    extra_prompt = getenv("CORAL_PROMPT_SYSTEM", "")

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"{{coral_instruction}} {SYSTEM_PROMPT} {extra_prompt} {{coral_messages}}"
            ),
            MessagesPlaceholder("history"),
        ]
    )

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

    history = InMemoryHistory()

    logger.info(f"Connecting to Coral @ '{CORAL_CONNECTION_URL}'")
    async with client.session("coral") as coral_session:
        try:
            coral_tools = await load_mcp_tools(
                coral_session,
                connection=client.connections["coral"],
            )
            other_connections = list(
                filter(lambda k: k != "coral", client.connections.keys())
            )
            other_tools = []
            for server in other_connections:
                other_tools.extend(await client.get_tools(server_name=server))
            logger.info(f"Found {len(other_tools)} non-coral tools.")
            tools = coral_tools
        except:
            logger.exception("Failed to get MCP tools")
            sys.exit(1)
        logger.info("Building chain...")
        chain = prompt | model.bind_tools(tools + [test_tool])

        logger.info("Building tool runner...")
        tool_runner = ToolRunner(tools)

        logger.info("Fetching instruction resource...")
        coral_instruction = (
            await load_mcp_resources(coral_session, uris="coral://agent/instruction")
        )[0]

        for _ in range(10):
            if claim_handler.no_budget():
                logger.info("No more budget - breaking loop")
                break
            coral_messages = (
                await load_mcp_resources(coral_session, uris="coral://messages")
            )[0]

            logger.debug("history = %s", history)

            logger.info("Making completion request...")
            step_result: BaseMessage = await chain.ainvoke(
                {  # We pass in our loaded resources here (as_string is safe here because we know these resources always return text)
                    "coral_instruction": [
                        SystemMessage(content=coral_instruction.as_string())
                    ],
                    "coral_messages": [
                        SystemMessage(content=coral_messages.as_string())
                    ],
                    "history": history.messages,
                }
            )

            history.add_message(step_result)

            if type(step_result) is AIMessage:
                tool_calls = step_result.tool_calls
                for call in tool_calls:
                    history.add_message(await tool_runner.run_tool(call))

            logger.debug("step_result: %s", step_result)


if __name__ == "__main__":
    asyncio.run(main())
