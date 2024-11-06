# app.py

import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler
from utils import MEMORY, load_document
from chat_with_documents import configure_retrieval_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot")

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx", "csv"])
if not uploaded_file:
    st.info("Please upload a document to start the chat.")
    st.stop()

# Save uploaded file to a temporary path for processing
temp_filepath = f"temp_dir/{uploaded_file.name}"
with open(temp_filepath, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Load and configure the retrieval chain
docs_or_agent = load_document(temp_filepath)
if isinstance(docs_or_agent, list):  # If documents are returned
    CONV_CHAIN = configure_retrieval_chain(docs_or_agent)
elif callable(docs_or_agent):  # If a CSV handler or agent is returned
    st.session_state.agent_executor = docs_or_agent

# Set up avatars for user and assistant
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
        # Run query through the configured CONV_CHAIN or CSV Agent Executor
        if uploaded_file.type == "text/csv" and st.session_state.agent_executor:
            # Directly use the .run method on the agent_executor for CSVs
            response = st.session_state.agent_executor.run(user_query)
        else:
            response = CONV_CHAIN.invoke(
                {"query": user_query, "chat_history": MEMORY.chat_memory.messages},
                callbacks=[stream_handler]
            )
            # Extract only the 'result' key for the main answer
            response = response.get("result", "I'm unable to process the query.")

        if response:
            container.markdown(response)
