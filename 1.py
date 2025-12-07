from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()

# ✅ PROPER Groq setup (FREE Llama 8B!)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# Chat history (your assignment!)
history = ChatMessageHistory()

# Agent chain
prompt = ChatPromptTemplate.from_template("Answer helpfully: {input}")
agent_chain = prompt | llm

# Add history automatically!
agent_with_history = RunnableWithMessageHistory(
    agent_chain,
    lambda: history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat_loop():
    print("🤖 Groq Llama 8B Chat (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break
        
        # Chat with HISTORY!
        response = agent_with_history.invoke(
            {"input": user_input},
            config={"configurable": {}}
        )
        
        print(f"Bot: {response.content}\n")

if __name__ == "__main__":
    chat_loop()