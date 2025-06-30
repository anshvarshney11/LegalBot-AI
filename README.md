🚀LegalBot – PDF Q&A Chatbot using LLMs
LegalBot is a document-aware AI chatbot that can answer user queries based on a given legal PDF. It uses LangChain, sentence-transformers, vector databases, and a locally downloaded LLM (distilgpt2) for efficient, offline retrieval-augmented generation (RAG).

Folder Structure
LegalBot/
│
├── app.py                     # Streamlit app
├── requirements.txt           # All dependencies
├── data/
│   └── AI Training Document.pdf    # Input document
├── models/
│   └── distilgpt2/                 # Locally stored distilgpt2 model
└── vectordb/                  # Chroma vector DB

Features
1. PDF parsing using PyPDF
2. Chunking using LangChain's RecursiveCharacterTextSplitter
3. Embedding via BAAI/bge-small-en-v1.5
4. Local vector store using Chroma
5. Question answering using a local distilgpt2 model
6. Clean UI built with Streamlit
7. Highlighted source text references

Chunking Logic:
1. Chunk size: 300 characters
2. Overlap: 50 characters
This ensures minimal context loss between chunks.

Embedding & Vector Store:
1. Embedding Model: BAAI/bge-small-en-v1.5 (compact + semantic)
2. Vector DB: Chroma (persistent local storage)

LLM Details:
1. Model: distilgpt2 (downloaded locally via huggingface_hub)
2. Generation parameters:
3. max_new_tokens=256
4. temperature=0.7
5. top_p=0.9
6. repetition_penalty=1.2

Prompt Format: Prompt = User question only (RAG handles context injection via LangChain's RetrievalQA chain)

👤 Made By:
Ansh Varshney
anshvarshney1109@gmail.com
