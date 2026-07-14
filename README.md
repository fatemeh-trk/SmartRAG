# 📚 TechHive

A local AI-powered document management and Retrieval-Augmented Generation (RAG) system built with **Python**, **Streamlit**, **Sentence Transformers**, and **ChromaDB**.

The project allows users to upload documents, split them into semantic chunks, generate vector embeddings, store them in ChromaDB, manage uploaded files, inspect generated chunks, and delete files or selected chunks through an interactive interface.

---

# ✨ Features

## Document Management

- Upload supported documents
- Automatic document chunking
- Generate embeddings using Sentence Transformers
- Store chunks inside ChromaDB

---

## File Management

- Display uploaded files
- Show file metadata
- Delete uploaded files
- Automatically remove all chunks belonging to deleted files

---

## Chunk Management

- Display all chunks of a selected document
- Show:
  - Chunk Index
  - Preview
  - Text Length
- Interactive chunk selection using Streamlit DataEditor
- Multi-select chunk deletion
- Automatic database refresh after deletion

---

# 🏗️ Project Architecture

```
                Upload Document
                       │
                       ▼
                Document Loader
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
               Embedding Generator
                       │
                       ▼
                  ChromaDB
                       │
      ┌────────────────┴───────────────┐
      │                                │
      ▼                                ▼
 File Management                Chunk Management
```

---

# 📂 Project Structure

```
SRC/
│
├── main.py                # Application entry point
├── streamlit_c.py         # User Interface (Streamlit)
├── functions.py           # Business logic
├── database.py            # Database utilities (if applicable)
├── ...
```

---

# 🧠 Technologies

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Pandas

---

# ⚙️ Current Workflow

```
Upload File
      │
      ▼
Chunking
      │
      ▼
Embedding
      │
      ▼
Store in ChromaDB
      │
      ▼
View Files
      │
      ▼
View Chunks
      │
      ▼
Select Chunks
      │
      ▼
Delete Selected Chunks
```

---

# 🚀 Current Capabilities

✅ Upload documents

✅ Generate semantic chunks

✅ Generate embeddings

✅ Store vectors in ChromaDB

✅ Display uploaded files

✅ Delete uploaded files

✅ Display chunks

✅ Multi-select chunk deletion

---

# 🛣️ Roadmap

## Completed

- [x] File upload
- [x] Chunk generation
- [x] Embedding generation
- [x] ChromaDB integration
- [x] File deletion
- [x] Chunk viewer
- [x] Multi-select chunk deletion

## Planned

- [ ] User authentication
- [ ] Admin/User roles
- [ ] Chunk editing
- [ ] Search inside documents
- [ ] Retrieval-Augmented Generation (RAG)
- [ ] LLM integration
- [ ] Conversation history
- [ ] Export results
- [ ] Docker support

---

# 💡 Design Principles

The project follows several software engineering principles:

- Separation of Concerns (SoC)
- Single Responsibility Principle (SRP)
- Modular Architecture
- Reusable Functions
- Clean UI / Business Logic separation

---

# 📌 Status

Current Version:

**v0.4**

Latest feature:

> Multi-select Chunk Management using Streamlit DataEditor.

---

# 📜 License

This project is under development for learning purposes.