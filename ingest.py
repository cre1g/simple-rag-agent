import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader("./docs", glob="./*.txt", loader_cls=TextLoader)

raw_docs = loader.load()

print(f"Loaded {len(raw_docs)} files")

#Specifies chunk parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

all_chunks = splitter.split_documents(raw_docs)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

Chroma.from_documents(all_chunks, embedding_model, persist_directory="./chroma_db")

print("Database created")