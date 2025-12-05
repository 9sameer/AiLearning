# simple_summarize.py
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# load .env so GOOGLE_API_KEY is available in os.environ
load_dotenv()

# --- 1) small summarizer LLM (Gemini 2.5 lite) ---
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1
)

# --- 2) simple chat history (strings already labeled Human / AI) ---
# In your real app replace this with the real stored history lines.
history = [
    "Human: What's AI?",
    "AI: AI stands for Artificial Intelligence...",
    "Human: Explain agents",
    "AI: An agent is a system that acts on behalf of a user...",
    "Human: LangChain?",
    "AI: LangChain is a library to build LLM apps...",
    "Human: History storage?",
    "AI: You can store history in memory, DB, or a message store.",
    "Human: Gemini SLM?",
    "AI: Gemini SLM is a smaller model good for quick tasks.",
    "Human: Free tier?",
    "AI: There is a free tier with limits.",
]

# --- 3) take last 10 messages and turn into one text block ---
last_10 = history[-10:]
context = "\n".join(last_10)

# --- 4) build a simple prompt that asks the SLM to summarize ---
prompt = ChatPromptTemplate.from_template(
    "Summarize this chat in 2-3 sentences:\n\n{context}"
)
summary_chain = prompt | summary_llm

# --- 5) call the SLM and print the summary ---
result = summary_chain.invoke({"context": context})
print(getattr(result, "content", str(result)))
