import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

#Starts a server, waits for request
#Searches DB, sends response, stores history each iteration

load_dotenv()

app = Flask(__name__)
CORS(app)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

#chat history via memory means it's shared between users
chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get("message")

    #Ensures continuity in similarity searches 
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Build a search query based on chat history and the user's question"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    contextualized_prompt = contextualize_q_prompt.format(question=query, chat_history=chat_history)
    contextualized = llm.invoke(contextualized_prompt)

    rag_data = chroma_db.similarity_search(contextualized.content, k=2)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in rag_data])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer based ONLY on context: {context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    final_prompt = prompt_template.format(context=context_text, question=query, chat_history=chat_history)
    response = llm.invoke(final_prompt)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))

    return jsonify({"answer": response.content})

if __name__ == '__main__':
    app.run(port=5000, debug=True)