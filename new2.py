import os
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory #stores messages of a chat session
from langchain_core.chat_history import BaseChatMessageHistory              #blueprint for storing chat messages.
from langchain_core.runnables.history import RunnableWithMessageHistory     #wraps an LLM so it can remember past messages
from langchain_core.prompts import ChatPromptTemplate    #builds prompts dynamically
from langchain_groq import ChatGroq
from dotenv import load_dotenv    #loads environment variables like GROQ_API_KEY

load_dotenv()

#Main agent 
your_agent_llm = ChatGroq(
    model="llama-3.3-70b-versatile",      
    temperature=0.1   #Low temperature = stable, controlled answers.
)
your_agent_prompt = ChatPromptTemplate.from_template("Answer: {input}")
your_agent_chain = your_agent_prompt | your_agent_llm

#This line wraps your LLM inside memory support.
    #Each user has a session_id
    #All messages in that session are saved in history_store
    #Next time the same session calls the model, the history is included
agent_with_history = RunnableWithMessageHistory(
    your_agent_chain,
    # This function tells LangChain: “Whenever a user chats, get their stored message history from history_store.
    # If there is no history yet, create a new empty ChatMessageHistory for this user.”
    # This is how each user gets their own memory.
    lambda session_id: history_store.setdefault(session_id, ChatMessageHistory()),
    input_messages_key="input", #“The incoming user message is located in the field called input.”
    history_messages_key="history", #When building the prompt for the LLM, put the conversation history in a field called history.”
)

#This smaller model is fast.
summary_llm = ChatGroq(
    model="llama-3.1-8b-instant",   
    temperature=0.1
)

# STORAGE (HISTORY STORAGE DICTIONARY)
history_store: Dict[str, BaseChatMessageHistory] = {}

#This ensures memory exists for each user.
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

# PART 3: SUMMARIZER (SAME)Takes the last 10 messages, Converts them into plain text (with Human/AI labels)
# Sends them to the 8B summarizer model , Returns a 2–3 sentence final summary

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


#TEST:Sends 12 questions to the chat agent.
#     Each response is stored as history.
#     After the loop:
#     Summarizes the last 10 chat messages
#     Prints final summary , Prints the total number of messages stored
#     This acts as a full test run to verify: LLM works , Memory works, Summarizer works

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

#This runs the test when the script is executed.
if __name__ == "__main__":
    demo()
