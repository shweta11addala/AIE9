"""An agent graph with a post-response fact-verification node.

After the agent responds, a structured evaluator scores confidence in the answer
and lists uncertain claims. If confidence is low, the agent is sent back to revise
with explicit feedback. A loop limit prevents infinite retries.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt

MAX_REVISIONS = 3


class FactCheckResult(BaseModel):
    confidence: int = Field(
        description="Confidence score 0-100 that all claims in the response are accurate"
    )
    uncertain_claims: list[str] = Field(
        description="List of specific claims that may be inaccurate or unverified"
    )


def _build_model_with_tools():
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    model = _build_model_with_tools()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def route_after_agent(state: MessagesState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "action"
    return "fact_check"


_fact_check_prompt = ChatPromptTemplate.from_template(
    "You are a rigorous fact-checker. Given a user query and an agent's response, "
    "score the confidence (0-100) that every factual claim in the response is accurate. "
    "Also list any specific claims that are uncertain, unverifiable, or potentially wrong.\n\n"
    "User Query:\n{query}\n\n"
    "Agent Response:\n{response}"
)


def fact_check_node(state: MessagesState) -> dict:
    # Count prior revision requests to enforce loop limit
    revision_count = sum(
        1 for m in state["messages"]
        if isinstance(m, HumanMessage) and m.content.startswith("FACT_CHECK_REVISION:")
    )
    if revision_count >= MAX_REVISIONS:
        return {"messages": [AIMessage(content="FACT_CHECK:DONE")]}

    query = state["messages"][0].content
    response = state["messages"][-1].content

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(FactCheckResult)
    result = (_fact_check_prompt | structured_model).invoke(
        {"query": query, "response": response}
    )

    if result.confidence >= 80 or not result.uncertain_claims:
        return {"messages": [AIMessage(content="FACT_CHECK:PASS")]}

    claims_text = "\n".join(f"- {c}" for c in result.uncertain_claims)
    feedback = (
        f"FACT_CHECK_REVISION: Your previous response scored {result.confidence}/100 "
        f"confidence. Please revise it to address these uncertain claims:\n{claims_text}"
    )
    return {"messages": [HumanMessage(content=feedback)]}


def fact_check_decision(state: MessagesState):
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    if content in ("FACT_CHECK:PASS", "FACT_CHECK:DONE"):
        return "end"
    return "revise"


def build_graph():
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("fact_check", fact_check_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"action": "action", "fact_check": "fact_check"},
    )
    graph.add_conditional_edges(
        "fact_check",
        fact_check_decision,
        {"revise": "agent", "end": END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
