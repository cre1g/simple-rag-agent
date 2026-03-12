# simple-rag-agent
A minimal Retrieval Augmented Generation (RAG) GPT chatbot built with Flask, LangChain, Chroma, and an HTML widget frontend.

<img width="1912" height="870" alt="landingpagewidget" src="https://github.com/user-attachments/assets/b1aac1f9-f9ca-4533-8e22-6bf063a4f11a" />

Building a functional chatbot with RAG and an example widget

Check requirements.txt for dependencies. Running locally with python version 3.12.10

**ingest.py** chunks and embeds the sample data in /docs and stores it in Chroma 

**query.py**:
starts a Flask server and waits for a request from chat.html
does a similarity search given the query and chat_history
builds the final response, again using context from previous steps
delivers it back to chat.html

**chat.html** hosts the widget, mock webpage, and script to call the flask server

docs contain AI generated sample data for testing RAG functionality.

This demo is extremely minimal and skips out on basic error handling, security, and other essential features.
This is not to be used for production
