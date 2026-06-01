# 🚀 TechHive - RAG Assistant for Company Documents

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-0.4-green.svg)](https://ollama.com)

## 📖 About The Project

TechHive is an intelligent document assistant that uses **RAG (Retrieval-Augmented Generation)** to answer questions about company documents. It runs 100% locally on your laptop - no API costs, no data leaving your computer.

### Built With
- **Streamlit** - Web UI
- **Ollama** - Local LLM (Phi-3)
- **ChromaDB** - Vector Database
- **Sentence Transformers** - Multilingual Embeddings

## ✨ Features

- ✅ Semantic search in company documents
- ✅ Local execution (privacy first)
- ✅  English support
- ✅ Chat history
- ✅ Source display for transparency

## 📋 Prerequisites

- Python 3.12+
- Ollama installed
- 8GB+ RAM (16GB recommended)

## 🚀 Quick Start

1. Clone the repository
bash
git clone https://github.com/YOUR_USERNAME/TechHive.git


2. Install dependencies
bash
pip install -r requirements.txt
3. Download LLM model
bash
ollama pull phi3:mini
4. Run the app
bash
cd TechHive
streamlit run SRC/rag_chat.py