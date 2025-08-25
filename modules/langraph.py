import os
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, add_messages, START
from langchain_core.messages import SystemMessage, BaseMessage
from pydantic import BaseModel
from typing import List, Annotated, Optional
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool

from dotenv import load_dotenv
load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")

class AgentState(BaseModel):
    """
    Represents the state of our agent's conversation.
    messages now correctly accepts a list of BaseMessage objects (Human or AI messages).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str = ""
    doc_id: Optional[str] = None
    last_action: str = ""

def build_agent_graph(tools: List[BaseTool]):
    """Builds the LangGraph with dynamic tool calling logic for query processing only."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    system_prompt = """
    You are an expert query processing agent for a RAG system. You have access to tools and must route queries to the correct workflow.

    CRITICAL: You can see the current context in the user's state. Use this information to make routing decisions.

    ROUTING RULES:

    1. DOCUMENT QUESTIONS (when doc_id is provided):
        - ANY question when doc_id exists should be treated as a document question
        - Workflow: retriever(user_question, doc_id) → qa_tool(question, chunks)
        - Examples: "What are the findings?", "Tell me about X", "Explain the methodology"

    2. DOCUMENT SUMMARY (when doc_id is provided and summary requested):
        - Trigger: Words like "summary", "summarize", "overview", "main points"
        - Workflow: retriever("main topics key findings summary overview", doc_id) → summarization(chunks, doc_id)

    3. WEATHER QUERIES:
        - Trigger: Words like "weather", "temperature", "forecast"
        - Workflow: get_weather(location)
        - Extract location from query, default to "Chennai"

    4. GENERAL WEB SEARCH (when NO doc_id):
        - Trigger: General knowledge questions without a loaded document
        - Workflow: tavily_search(user_query)

    IMPORTANT EXECUTION RULES:
    - NEVER ask the user for doc_id - it's provided in the context
    - If doc_id exists, treat the question as document-related unless it's clearly about weather
    - Complete the full workflow - call the second tool after getting results from the first
    - If retriever finds no chunks, inform user that the document doesn't contain relevant information

    AVAILABLE TOOLS:
    {tools}

    Remember: The doc_id is already provided in the state - use it directly without asking the user.
    """
    
    tools_str = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    formatted_prompt = system_prompt.format(tools=tools_str)
    
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """The main agent function to decide on the next step."""
        # Create context-aware system message
        doc_status = f"LOADED: {state.doc_id}" if state.doc_id else "NO DOCUMENT"
        context_message = f"""
CURRENT CONTEXT:
- Document Status: {doc_status}
- User Query: "{state.user_query}"

Based on this context, route the query appropriately. If a document is loaded, treat questions as document-related unless they're clearly about weather or external topics.
"""
        
        # Add context to system prompt
        full_system_prompt = formatted_prompt + context_message
        
        messages = [SystemMessage(content=full_system_prompt)] + state.messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    workflow = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())