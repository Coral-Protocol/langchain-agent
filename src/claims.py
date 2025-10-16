from os import getenv
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
