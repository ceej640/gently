"""
Agent Core — The reasoning engine.

This module contains the LLM-based reasoning:
- prompt.py: Prompt construction for different thinking modes
- reasoning.py: Agent.think() that calls the LLM
- parsing.py: Parse LLM response into structured actions
"""

from .prompt import build_agent_prompt, format_context, format_world_state, format_trigger_context
from .reasoning import Agent, create_think_function
from .parsing import parse_agent_response

__all__ = [
    "build_agent_prompt",
    "format_context",
    "format_world_state",
    "format_trigger_context",
    "Agent",
    "create_think_function",
    "parse_agent_response",
]
