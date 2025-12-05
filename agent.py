# simple_history_summarizer.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# ----------------------------------------
# 1) REAL CHAT HISTORY STORAGE (same as your code)
# ----------------------------------------
history_store = {}

# If history doesn't exist, create it
def get_history(user_id):
    if user_id not in history_store:
        history_store[user_id] = ChatMessageHistory()
    return history_store[user_id]

# ----------------------------------------
# 2) ADD SOME CHAT MESSAGES (simulate real usage)
# ----------------------------------------
user_id = "student123"
history = get_history(user_id)

# Normally your agent fills this automatically.
# For simplicity, we simulate:
history.add_user_message("What is AI?")
history.add_ai_message("AI is Artificial Intelligence.")

history.add_user_message("Explain agents.")
history.add_ai_message("Agents are systems that act on behalf of users.")

history.add_user_message("What is LangChain?")
history.add_ai_message("LangChain is a framework to build AI applications.")

history.add_user_message("What is memory in agents?")
history.add_ai_message("Memory stores previous interactions.")

history.add_user_message("What is Gemini SLM?")
history.add_ai_message("SLM is a smaller model for fast tasks.")

# ----------------------------------------
# 3) TAKE LAST 10 MESSAGES FROM REAL HISTORY
# ----------------------------------------
last_10_objects = history.messages[-10:]  # these are message objects
last_10_text = []

for msg in last_10_objects:
    role = "Human" if msg.type == "human" else "AI"
    last_10_text.append(f"{role}: {msg.content}")

context = "\n".join(last_10_text)

# ----------------------------------------
# 4) USE SLM TO SUMMARIZE THE REAL CHAT HISTORY
# ----------------------------------------
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template(
    "Summarize this chat in 2–3 sentences:\n\n{context}"
)

chain = prompt | summary_llm
result = chain.invoke({"context": context})

print("\n====== SUMMARY OF LAST 10 MESSAGES ======")
print(result.content)
