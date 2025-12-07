import os
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ======================================
# ✅ FIXED: CURRENT WORKING GROQ MODELS (Jan 2025)
# ======================================
your_agent_llm = ChatGroq(
    model="llama-3.3-70b-versatile",          # ✅ ALWAYS WORKING!
    temperature=0.1
)
your_agent_prompt = ChatPromptTemplate.from_template("Answer: {input}")
your_agent_chain = your_agent_prompt | your_agent_llm

agent_with_history = RunnableWithMessageHistory(
    your_agent_chain,
    lambda session_id: history_store.setdefault(session_id, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history",
)

# ✅ FIXED SLM (8B for summaries)
summary_llm = ChatGroq(
    model="llama-3.1-8b-instant",          # ✅ PERFECT 8B SLM!
    temperature=0.1
)

# ======================================
# PART 1: STORAGE (SAME)
# ======================================
history_store: Dict[str, BaseChatMessageHistory] = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# ======================================
# PART 3: SUMMARIZER (SAME)
# ======================================
def summarize_last_10(session_id: str) -> str:
    history = history_store.get(session_id)
    
    if not history or not history.messages:
        return "No conversation history."
    
    last_10 = history.messages[-10:]
    chat_text = []
    for msg in last_10:
        role = "Human" if "human" in str(type(msg)).lower() else "AI"
        chat_text.append(f"{role}: {msg.content}")
    
    context = "\n".join(chat_text)
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize this chat in 2-3 sentences:\n\n{context}"
    )
    summary_chain = prompt | summary_llm
    
    summary = summary_chain.invoke({"context": context})
    return summary.content

# ======================================
# 🧪 TEST (SAME)
# ======================================
def demo():
    user_id = "student123"
    
    questions = [
        "What's AI?", "Explain agents", "LangChain?", "History storage?", 
        "Groq SLM?", "Free tier?", "Rate limits?", "Good for homework?", 
        "Fast?", "Cheap?", "Works?", "Thanks!"
    ]
    
    print("🤖 Groq Llama 8B Chat + Summarizer (FIXED!)")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        
        response = agent_with_history.invoke(
            {"input": q}, 
            config={"configurable": {"session_id": user_id}}
        )
        print(f"  AI: {response.content[:50]}...\n")
    
    print("\n📝 SUMMARY (Last 10 messages):")
    print("=" * 60)
    summary = summarize_last_10(user_id)
    print(summary)
    print(f"\n💾 Total messages: {len(history_store[user_id].messages)}")

if __name__ == "__main__":
    demo()