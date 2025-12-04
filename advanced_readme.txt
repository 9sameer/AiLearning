🚀 Agent with Conversation History + Last-10 Message Summarizer (Overview)

This project demonstrates a simple conversational AI agent built using Google Gemini and LangChain, combined with a session-based memory system and a small language model (SLM) summarizer.
It provides a clear example of how chat history can be captured and summarized in AI-driven applications.

🔧 What This Script Does

Loads environment variables using .env

Stores per-user conversation history in memory

Builds an LLM pipeline using Gemini

Wraps the agent with RunnableWithMessageHistory to attach session memory

Extracts and summarizes the last 10 messages with a lightweight summarizer model

Simulates a full conversation flow using a demo function

🧱 Core Components (Explained Simply)
1. Environment Setup

The script uses python-dotenv to load GOOGLE_API_KEY from a .env file, keeping secrets secure and out of source control.

2. Conversation Memory Storage

A dictionary stores history for each user session:

Keys = session IDs

Values = ChatMessageHistory objects

Each history contains the full list of messages in order

This enables multi-turn conversations and later summarization.

3. The Agent Pipeline

The agent consists of:

A simple prompt template that formats user input

A Gemini model (gemini-2.0-flash-exp) that generates answers

A pipeline operator linking prompt → LLM

The agent does not use conversation history for generating responses—the summarizer does.

4. Adding Memory With RunnableWithMessageHistory

The pipeline is wrapped so that every call:

Retrieves the session’s history

Injects it into the agent inputs

This allows you to associate every response with a specific user.

Note: Since the prompt does not include {history}, the agent receives history but does not use it in its responses.

5. Last-10 Message Summarizer

Your assignment-focused function performs:

Fetching the last 10 messages from a user’s history

Converting them into readable text (Human/AI labels)

Passing them to a small Gemini model (gemini-2.5-flash-lite)

Producing a 2–3 sentence summary

This demonstrates how SLMs are ideal for fast, inexpensive summarization.

6. Demo Simulation

The demo:

Creates a session

Sends 12 questions to the agent

Lets the agent answer each one

Prints a summary of the last 10 messages

Shows how many total messages were stored

A typical result is 24 messages (12 Human + 12 AI).

📊 How the Flow Works
User Input → Agent → AI Response → Save to History → Repeat
                              ↓
                   Summarize Last 10 Messages

⚠️ Important Notes

History is stored but not used by the agent when generating responses
(your prompt does not include {history})

Some LangChain versions auto-save messages; some don’t

Summarizer only works if the history contains messages

Handle .content carefully — some models return plain strings

🔮 What You Can Build on Top

Memory-aware chatbots

Summaries for long conversations

Interactive assistant workflows

History-powered RAG pipelines

Session analytics dashboards