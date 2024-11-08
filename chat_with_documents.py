# chat_with_documents.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os


def configure_retrieval_chain(chunked_docs):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-3.5-turbo")
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )
    return retrieval_chain
