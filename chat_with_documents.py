# chat_with_documents.py

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def configure_retrieval_chain(chunked_docs):
    # Create embeddings and vector store from chunked documents
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Initialize OpenAI model and Retrieval Chain
    model = ChatOpenAI(model="gpt-3.5-turbo")
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True
    )
    return retrieval_chain
