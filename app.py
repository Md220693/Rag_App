# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from streamlit.external.langchain import StreamlitCallbackHandler
from utils import MEMORY, load_document
from chat_with_documents import configure_retrieval_chain
#djkffsflsdfklklsdfl;sdfklsdfkl
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Make sure to set it in your environment variables.")
else:
  
    os.environ["OPENAI_API_KEY"] = openai_api_key

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "csv"])
if not uploaded_file:
    st.info("Please upload a document to start the chat.")
    st.stop()

temp_filepath = f"temp_dir/{uploaded_file.name}"
with open(temp_filepath, "wb") as f:
    f.write(uploaded_file.getbuffer())


docs_or_agent = load_document(temp_filepath)
if isinstance(docs_or_agent, list): 
    CONV_CHAIN = configure_retrieval_chain(docs_or_agent)
elif callable(docs_or_agent): 
    st.session_state.agent_executor = docs_or_agent

avatars = {"human": "user", "ai": "assistant"}

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("Hello! Ask me anything about the uploaded document.")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assistant = st.chat_message("assistant")
if user_query := st.chat_input(placeholder="Type your query/message here!"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)

    with st.chat_message("assistant"):
        if uploaded_file.type == "text/csv" and st.session_state.agent_executor:
            response = st.session_state.agent_executor.run(user_query)
        else:
            response = CONV_CHAIN.invoke(
                {"query": user_query, "chat_history": MEMORY.chat_memory.messages},
                callbacks=[stream_handler]
            )
            response = response.get("result", "I'm unable to process the query.")

        if response:
            container.markdown(response)
