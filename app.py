import streamlit as st
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

# Your existing manual assistant code

# Function to handle document upload and embedding
def handle_document_upload(uploaded_files):
    if uploaded_files:
        # Perform document embedding and store it in Vector_db
        pdf_file_path = save_uploaded_file(uploaded_files)
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        all_documents = text_splitter.split_documents(documents)

        hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        Vector_db = Chroma.from_documents(collection_name="document_docs", documents=all_documents, embedding=hf_embed,
                                         persist_directory="./Persist_dir")

        Vector_db.persist()
        st.success("Document uploaded and embedded successfully!")

# Function to save the uploaded file and return the file path
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to display the result using Streamlit
def display_result(question, result, similar_docs):
    result_html = f"<h1>{question}</h1>"
    result_html += f"<p>{result}</p>"
    result_html += "<p><hr/></p>"
    for d in similar_docs:
        source_id = d.metadata["source"]
        page = d.metadata['page']
        result_html += f"<p>{d.page_content}<br/>(Source: {source_id} Page: {page})</p>"
    st.markdown(result_html, unsafe_allow_html=True)

# Streamlit UI components
st.title("Document Upload and Question-Answering")

uploaded_files = st.file_uploader("Upload your document (PDF)", type=["pdf"], key="doc_upload")

if st.button("Upload and Embed Document"):
    handle_document_upload(uploaded_files)

# Streamlit UI component for user question input
user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    # Perform question-answering using your existing code
    similar_docs = get_similar_docs(user_question, similar_doc_count=4)
    conversation = "".join(conv["user_question"] + "\n" + conv["bot_response"] for conv in conversation_history)
    result = qa_chain({"input_documents": similar_docs, "conversation_history": conversation, "question": user_question})

    if result is None:
        st.error("Couldn't find data from the document.")
    else:
        conversation_history.append({"user_question": user_question, "bot_response": result["output_text"]})
        while len(conversation_history) != 1:
            conversation_history.pop(0)

        display_result(user_question, result["output_text"], similar_docs)
