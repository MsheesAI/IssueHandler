"""LangGraph ReAct agent for GitHub issue management powered by Groq and FastMCP."""

import os
import sys
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = (
    "You are an expert GitHub Issue Assistant.\n"
    "You help users manage GitHub issues through natural language conversations.\n\n"
    "Your core capabilities:\n"
    "1. READ ISSUES: list issues with `list_issues`, count them with `count_issues`, "
    "or fetch one with `get_issue`.\n"
    "2. EXPLAIN ISSUES: fetch the issue and its comments with `get_issue`, then explain:\n"
    "   - a plain-language summary of what is reported\n"
    "   - the probable root cause\n"
    "   - suggested next steps or code-fix recommendations\n"
    "3. CREATE ISSUES: write a clean title and a structured markdown body, then call "
    "`create_issue`. Use `search_issues` first to check for duplicates.\n"
    "4. CLOSE ISSUES: use `close_issue`, only when the user explicitly asks to close an issue.\n\n"
    f"The default repository is '{settings.default_repo or 'None'}'. When the user says "
    "'my repo', 'current repo' or 'default repo', or names no repository, omit the `repo` "
    "argument of the tools so the default is used. If there is no default and no repo was "
    "given, ask the user for it in 'owner/repo' format.\n"
    "If a tool returns an error, explain it to the user in plain words. "
    "Never invent repository names or issue numbers.\n"
    "Format responses in clean, readable Markdown."
)

_mcp_client: Optional[MultiServerMCPClient] = None
_mcp_tools: dict[str, Any] = {}


async def get_mcp_tools() -> dict[str, Any]:
    global _mcp_client, _mcp_tools
    if not _mcp_tools:
        proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ)
        env["PYTHONPATH"] = proj_dir

        _mcp_client = MultiServerMCPClient(
            {
                "github": {
                    "command": sys.executable,
                    "args": ["-m", "app.mcp_server"],
                    "transport": "stdio",
                    "env": env,
                    "cwd": proj_dir,
                }
            }
        )
        tools_list = await _mcp_client.get_tools()
        _mcp_tools = {tool.name: tool for tool in tools_list}
    return _mcp_tools


def get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )


async def get_agent(checkpointer=None):
    tools_dict = await get_mcp_tools()
    tools = list(tools_dict.values())
    llm = get_llm()
    saver = checkpointer if checkpointer is not None else MemorySaver()

    # Tool failures go back to the model as text instead of crashing the request
    tool_node = ToolNode(tools, handle_tool_errors=lambda e: f"Tool failed: {e}")

    return create_react_agent(
        llm,
        tool_node,
        checkpointer=saver,
        prompt=SYSTEM_PROMPT,
    )