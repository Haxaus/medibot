import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv  # For local development

# Load environment variables locally (optional)
load_dotenv()

DB_FAISS_PATH = "vector_db/"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don’t know the answer, just say that you don’t know, don’t try to make up an answer. 
    Don’t provide anything out of the given context.

    Context: {context}
    Question: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def main():
    st.title("Hi!! I am Medibot.\nAsk me anything about the medical field")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        # Fetch token securely: Streamlit Cloud secrets or local env
        HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
        if not HF_TOKEN:
            st.error("Hugging Face token not found. Please set HF_TOKEN in secrets or environment.")
            return

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()