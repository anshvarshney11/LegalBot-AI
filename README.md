🚀LegalBot – PDF Q&A Chatbot using LLMs
LegalBot is a document-aware AI chatbot that can answer user queries based on a given legal PDF. It uses LangChain, sentence-transformers, vector databases, and a locally downloaded LLM (distilgpt2) for efficient, offline retrieval-augmented generation (RAG).




Folder Structure
bash
Copy
Edit
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
PDF parsing using PyPDF

Chunking using LangChain's RecursiveCharacterTextSplitter

Embedding via BAAI/bge-small-en-v1.5

Local vector store using Chroma

Question answering using a local distilgpt2 model

Clean UI built with Streamlit

Highlighted source text references


Chunking Logic
Chunk size: 300 characters

Overlap: 50 characters
This ensures minimal context loss between chunks.

Embedding & Vector Store
Embedding Model: BAAI/bge-small-en-v1.5 (compact + semantic)

Vector DB: Chroma (persistent local storage)
 



LLM Details
Model: distilgpt2 (downloaded locally via huggingface_hub)

Generation parameters:

max_new_tokens=256

temperature=0.7

top_p=0.9

repetition_penalty=1.2



Prompt Format
Prompt = User question only (RAG handles context injection via LangChain's RetrievalQA chain)


👤 Made By
Ansh Varshney
anshvarshney1109@gmail.com