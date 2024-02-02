import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import time
import torch

st.set_page_config(page_title='MedDoc-Bot', page_icon='plri.png', initial_sidebar_state='collapsed')

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I'd be happy to help you interpret the uploaded document. 🤖"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hallo!"]

def conversation_chat(query, chain, history, model_path):
    start_time = time.time()
    result = chain({"question": query, "chat_history": history})
    end_time = time.time()
    response_time = end_time - start_time
    history.append((query, result["answer"], model_path, response_time))
    return result["answer"], response_time

def display_chat_history(chain, model_path):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output, response_time = conversation_chat(user_input, chain, st.session_state['history'], model_path)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

            st.info(f"Response generated using {model_path} in {response_time:.2f} seconds")

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(model_path, vector_store):
    llm = LlamaCpp(
        streaming=True,
        model_path=model_path,
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def login():
    # Logo
    st.image("LOGO.png", use_column_width=True)  # replace logo
    st.title("Login to Access the MedDoc-Bot")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    login_button = st.button("Login")

    if login_button:
        # Perform authentication logic here (e.g., check username and password)
        if username == "User" and password == "User@123":
            st.success("Login successful!")
            st.session_state.login_success = True
            st.rerun()
        else:
             st.error("Invalid username or password!") # Redirect to the main page

def main():
    # Check if the user is logged in
    if 'login_success' not in st.session_state or not st.session_state.login_success:
        login()
    else:
        # Initialize session state
        initialize_session_state()

        # Logo
        st.image("MHH_LOGO.png", use_column_width=True)  # replace logo

        st.title("Select Model and Chat with Uploaded Medical PDF")

        # Choose model using dropdown
        model_option = st.sidebar.selectbox("Select Model", ["llama-2-13b.Q5_K_S", "medalpaca-13b", "meditron-7b.Q5_K_S", "mistral-7b-instruct"])

        # Model paths dictionary
        model_paths = {
            "llama-2-13b.Q5_K_S": "llama-2-13b.Q5_K_S.gguf",  # replace "llama-2-13b.Q5_K_S.gguf"
            "medalpaca-13b": "medalpaca-13b.Q5_K_S.gguf",  # replace "medalpaca-13b.Q5_K_S.gguf"
            "meditron-7b.Q5_K_S": "meditron-7b.Q5_K_S.gguf",  # replace "medicine-llm-13b.Q5_K_S.gguf"
            "mistral-7b-instruct": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",  # replace "mistral-7b-instruct-v0.1.Q4_K_M.Q5_K_S.gguf"
        }

        model_path = model_paths[model_option]

        # Initialize Streamlit
        st.sidebar.title("Document Processing")
        uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                              model_kwargs={'device': 'cuda'})

            # Create vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # Create the chain object
            chain = create_conversational_chain(model_path, vector_store)

            display_chat_history(chain, model_path)

if __name__ == "__main__":
    main()
