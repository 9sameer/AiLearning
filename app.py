import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

# ✅ Fixed model name
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Updated model name
    temperature=0.5
)

if "flowmessages" not in st.session_state:
    st.session_state["flowmessages"] = [
        SystemMessage(content="You are a comedian AI assistant")
    ]

def get_chatmodel_response(question):
    st.session_state["flowmessages"].append(HumanMessage(content=question))
    answer = chat.invoke(st.session_state["flowmessages"])
    st.session_state["flowmessages"].append(AIMessage(content=answer.content))
    return answer.content

user_input = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

if submit and user_input.strip() != "":
    with st.spinner("Thinking..."):
        response = get_chatmodel_response(user_input)
        st.subheader("The Response is")
        st.write(response)