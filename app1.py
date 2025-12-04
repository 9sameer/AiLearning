import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
st.title("🤖 Gemini Chat")

chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = chat.invoke(prompt)
    st.chat_message("assistant").write(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})