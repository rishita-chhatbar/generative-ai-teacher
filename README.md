# Generative AI Teacher

A Gemini-style Generative AI Teacher web application built with Streamlit, LangChain, and Groq to explain AI, Python, Machine Learning, and LLM concepts interactively.

## Features
- **Conversational AI Teacher:** Maintains chat history for natural follow-up questions and simple explanations.
- **Gemini-Style Input Bar (`+` Menu & Attachments):** Supports attaching files/photos, capturing photos via camera, and clearing chat history with one click.
- **Prompt & Chain Experiments:** Includes a Jupyter Notebook (`Open AIme.ipynb`) demonstrating prompts, `ChatPromptTemplate`, LangChain runnables (`|`), and output parsers.

## Project Structure
```text
generative-ai-teacher/
├── app.py             # Streamlit web app with LangChain + Groq integration
├── Open AIme.ipynb    # Notebook with LangChain prompt, chain, and LLM experiments
└── README.md          # Project documentation
