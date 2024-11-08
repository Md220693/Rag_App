# utils.py

import os
import pathlib
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.chat_models import ChatOpenAI
from zipfile import BadZipFile


def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

MEMORY = init_memory()

def load_document(temp_filepath):
    ext = pathlib.Path(temp_filepath).suffix.lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(temp_filepath)
        docs = loader.load()
    elif ext == ".docx":
        try:
            loader = Docx2txtLoader(temp_filepath)
            docs = loader.load()
        except BadZipFile:
            raise ValueError("The uploaded DOCX file is corrupted or not a valid DOCX file.")
    elif ext == ".csv":
        return load_csv_agent(temp_filepath)  # Use CSV-specific agent
    else:
        raise ValueError("Unsupported file type.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return text_splitter.split_documents(docs)

def load_csv_agent(csv_path):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),temperature=0.5)
    agent_executor = create_csv_agent(
        llm=llm,
        path=csv_path,
        verbose=False,
        allow_dangerous_code=True
    )
    return agent_executor
