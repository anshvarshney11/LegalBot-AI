import streamlit as st
from huggingface_hub import login
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os
from pypdf import PdfReader

# --- CONFIG ---
st.set_page_config(page_title="LegalBot Chat", layout="wide")
pdf_path = "data/AI Training Document.pdf"
db_folder = "vectordb"

# --- SIDEBAR ---
with st.sidebar:
    st.title("📄 LegalBot")
    st.markdown("Ask questions based on the uploaded legal PDF.")
    st.markdown("🔗 Made by **Ansh Varshney**")
    st.markdown("[Connect on LinkedIn](https://www.linkedin.com/in/anshvarshney-4b6466278///)")

# --- Loading PDF Text ---
@st.cache_data
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- Chunking ---
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_text(text)

# --- Embedding & Vector DB ---
def embed_and_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectordb = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=db_folder)
    vectordb.persist()
    return vectordb

# --- Loading LLM from Local Path ---
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="C:/hf_models/distilgpt2",           
        tokenizer="C:/hf_models/distilgpt2",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- Main App Logic ---
def main():
    st.header("🧠 LegalBot – Ask About the PDF")

    # Loading PDF
    with st.spinner("📖 Loading document..."):
        raw_text = load_pdf_text(pdf_path)
        chunks = chunk_text(raw_text)

    # Embed & store
    with st.spinner("🔍 Generating embeddings..."):
        vectordb = embed_and_store(chunks)

    # Loading LLM
    with st.spinner("🤖 Loading LLM..."):
        llm = load_llm()

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    # User Query
    query = st.text_input("💬 Ask a question about the document:")
    if query:
        with st.spinner(" Thinking..."):
            result = qa_chain({"query": query})

            st.markdown("### ✅  Answer:")
            st.write(result["result"])  

            st.markdown("---")
            st.markdown("### 📄 Sources:")
            for doc in result["source_documents"]:
                st.markdown(f"> {doc.page_content[:300]}...")

if __name__ == "__main__":
    main()