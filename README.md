# 🧠 SmartRAG - Intelligent Document Assistant

**RAG-based document Q&A system with local LLM (Ollama + ChromaDB + Streamlit)**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-0.4-green.svg)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 About The Project

SmartRAG is a production-ready **Retrieval-Augmented Generation (RAG)** system that answers questions from your company documents. It runs **100% locally** on your laptop - no API costs, no data leaving your computer.

### ✨ Features

- 🔍 **Semantic search** in documents (PDF, DOCX, TXT)
- 🤖 **Local LLM** support (Phi-3)
- 📚 **Vector database** with ChromaDB
- 🌐 **Multilingual** ( English)
- 🗑️ **Document management** (add/delete chunks)
- 🖥️ **Beautiful UI** with Streamlit
- 🔐 **Privacy-first** (100% local execution)

### 🛠️ Built With

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI |
| **Ollama** | Local LLM |
| **ChromaDB** | Vector Database |
| **Sentence Transformers** | Embeddings |
| **LangChain** | Text splitting |
| **PyPDF2 / python-docx** | Document parsing |

---

## 📋 Prerequisites

- **Python 3.12+**
- **Ollama** installed ([download](https://ollama.com))
- **8GB+ RAM** (16GB recommended)
- **Git** (optional, for cloning)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/fatemeh-trk/SmartRAG.git
cd SmartRAG
2. Create virtual environment (recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies

pip install -r requirements.txt
4. Download LLM model

ollama pull phi3:mini
# Or for better accuracy:
ollama pull gemma2:2b
5. Run the application

# Using main.py (recommended)
python -m SRC.main

# Or directly with Streamlit
streamlit run SRC/main.py

##🎯How It Works
text
User Question → Embedding → Vector Search → Relevant Chunks → LLM → Answer
Upload your documents (PDF, DOCX, TXT)

1.Chunking splits text into 250-character pieces

2.Embedding converts chunks to vectors

3.Search finds semantically similar chunks

4.LLM generates answers based on retrieved context

##📊 Usage Examples
Question	Expected Answer
Who is the CEO?	Engineer Ahmadi
When was the company founded?	2016
What are the working hours?	Saturday to Wednesday: 9 AM to 5 PM
Where is the headquarters?	Tehran, Valiasr Street
##🔧 Configuration
Change LLM model
In src/streamlit.py, change:

python
model='gemma2:2b'  # instead of 'phi3'

Adjust chunk size
In src/functions.py, modify:

python
chunk_size=250,  # characters per chunk
chunk_overlap=50,  # overlap between chunks
##🗑️ Database Management
The sidebar provides:

Delete by count - Remove last N chunks

Delete by ID - Remove specific chunks

Delete all - Clear entire database

🤝 Contributing
Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

##📞 Contact
GitHub
Project Link: https://github.com/fatemeh-trk/SmartRAG

##🙏 Acknowledgments
Streamlit for amazing UI framework

Ollama for local LLM runtime

ChromaDB for vector database

Sentence Transformers for embeddings

##📄 License
Distributed under the MIT License. See LICENSE for more information.

##⭐ Star History
If you find this project useful, please give it a star! ⭐