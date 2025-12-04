import os
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
# 1. FREE API KEY (get from https://aistudio.google.com/app/apikey)
# os.environ["GOOGLE_API_KEY"] = "paste-your-free-key-here"

# ======================================
# PART 1: YOUR EXISTING AGENT STORAGE
# ======================================
# This is where your agent stores history (replace with your storage)
history_store: Dict[str, BaseChatMessageHistory] = {}   
# Type - "dictionary with user IDs → chat histories"
#  {} Value - "empty database"


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """📦 Gets history for a user (your agent uses this)"""
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# ======================================
# PART 2: WRAP YOUR EXISTING AGENT
# ======================================
# Replace 'your_agent_chain' with your actual agent
your_agent_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")  # FREE
your_agent_prompt = ChatPromptTemplate.from_template("Answer: {input}")
your_agent_chain = your_agent_prompt | your_agent_llm  # ← Your agent pipeline The pipe (|) means output of the prompt step becomes input to the LLM step.

# Add history to your agent (ONE LINE!)
agent_with_history = RunnableWithMessageHistory(
    your_agent_chain,
    get_chat_history,  # ← Uses storage above
    input_messages_key="input",
    history_messages_key="history",
)

# ======================================
# PART 3: SUMMARIZER FUNCTION (YOUR ASSIGNMENT!)
# ======================================
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # ✅ FREE 8B SLM
    temperature=0.1
)

def summarize_last_10(session_id: str) -> str:
    """
    🎯 YOUR ASSIGNMENT FUNCTION!
    Gets last 10 messages → summarizes with SLM
    """
    history = history_store.get(session_id)
    
    if not history or not history.messages:
        return "No conversation history."
    
    # 📋 Step 1: Get LAST 10 messages
    last_10 = history.messages[-10:]
    
    # 📝 Step 2: Format like "Human: Hi → AI: Hello"
    chat_text = []
    for msg in last_10:
        role = "Human" if "human" in str(type(msg)).lower() else "AI"
        chat_text.append(f"{role}: {msg.content}")
    
    context = "\n".join(chat_text)
    
    # 🤖 Step 3: SLM summarizes (Gemini 8B)
    prompt = ChatPromptTemplate.from_template(
        "Summarize this chat in 2-3 sentences:\n\n{context}"
    )
    summary_chain = prompt | summary_llm
    
    summary = summary_chain.invoke({"context": context})
    return summary.content

# ======================================
# 🧪 TEST IT (Your assignment demo)
# ======================================
def demo():
    user_id = "student123"
    
    # print("🤖 Testing your agent + summarizer...")    
    # Simulate 12 chat turns (builds history)
    questions = [
        "What's AI?", "Explain agents", "LangChain?", "History storage?", 
        "Gemini SLM?", "Free tier?", "Rate limits?", "Good for homework?", 
        "Fast?", "Cheap?", "Works?", "Thanks!"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        
        
        # Your agent responds (saves to history automatically!)
        response = agent_with_history.invoke(
            {"input": q}, 
            config={"configurable": {"session_id": user_id}}    #3. This part connects identity to conversation memory,
            # This tells LangChain: “Use the history bucket for user → student123”
        )
        print(f"  AI: {response.content[:50]}...\n")
    
    # 🎯 YOUR ASSIGNMENT: Summarize last 10!
       
    summary = summarize_last_10(user_id)
    print(summary)
    print(f"\n💾 Total messages stored: {len(history_store[user_id].messages)}")
    
# What does demo() do?
# Starts a session with user ID "student123".
# Sends 12 questions to your agent.
# Agent replies each time AND saves messages automatically.
# At the end, calls your summarize_last_10(user_id) function.
# Prints the summary of the last 10 chat exchanges.
# Shows total stored messages.

if __name__ == "__main__":
    demo()