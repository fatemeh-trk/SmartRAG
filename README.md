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

- 📄 PDF Upload
- 🧠 Local RAG Pipeline
- 💾 Persistent ChromaDB Storage
- 🔍 Database Inspector
- 📊 Pandas Debug Table
- 🤖 Phi-3 Mini GGUF
- Added Metadata support for each chunk
- Added Database Inspector
- Display file name, file id, chunk index and upload time
- Improved database debugging

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
## Project Progress

### Completed

- [x] PDF Upload
- [x] Text Chunking
- [x] Embedding Generation
- [x] ChromaDB Integration
- [x] Persistent ChromaDB Storage

### In Progress

- [ ] Metadata
- [ ] Citation
- [ ] Streaming Response
- [ ] Hybrid Search
- [ ] Chat Memory

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