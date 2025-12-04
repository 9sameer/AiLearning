# === FIXED VERSION - Compatible with latest packages ===
import os
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state: AgentState):
    last_message = state["messages"][-1].content.lower()
    responses = {
        "weather": "Sunny, 75°F today.",
        "meeting": "2pm meeting confirmed for tomorrow.",
        "report": "Sales up 15% Q3. Full breakdown ready.",
        "flight": "NYC flight booked - business class.",
        "email": "Email to Sarah sent (Q3 template, CC John).",
        "expense": "Expense report pending accounting review.",
        "default": "Got it! What else can I help with?"
    }
    response = responses.get(last_message, responses["default"])
    return {"messages": [AIMessage(content=response)]}

# Compile agent
checkpointer = MemorySaver()
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
agent_graph = builder.compile(checkpointer=checkpointer)

# Simulate conversation
print("🔄 SIMULATING CONVERSATION...")
thread_id = "test-gemini-fixed"
config = {"configurable": {"thread_id": thread_id}}

messages = [
    "weather today?", "meeting tomorrow?", "sales report?", "flight nyc?", 
    "email sarah", "expense status?", "thanks!"
]

for i, msg in enumerate(messages, 1):
    print(f"Turn {i}: {msg}")
    agent_graph.invoke({"messages": [HumanMessage(content=msg)]}, config)

# Gemini summarizer
def summarize_with_gemini(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    history = list(agent_graph.get_state_history(config))
    
    recent_msgs = []
    for checkpoint in history[-10:]:
        msgs = checkpoint.values.get("messages", [])
        for msg in msgs[-2:]:
            recent_msgs.append(msg)
    
    conv_text = "\n".join([f"{'USER' if m.type=='human' else 'AI'}: {m.content}" for m in recent_msgs[-12:]])
    
    slm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-exp",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )
    
    prompt = f"Summarize in 3 bullets:\n```\n{conv_text}\n```"
    summary = slm.invoke(prompt).content
    return summary

# Run!
print("\n🤖 GEMINI SUMMARY:")
print("-" * 40)
print(summarize_with_gemini(thread_id))