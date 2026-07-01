from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
import tempfile
import streamlit as st
import ollama
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap= 50,
        separators=[
            "\n\n",     
            "\n",       
            "。",      
            "！",      
            "？",
            "،",       
            " ",        
            ""
            ],
        length_function = len,
        is_separator_regex = False
        )
    chunks = splitter.split_text(text)
    return chunks



def read_uploaded_file(uploaded_file):
    file_extension= Path(uploaded_file.name).suffix.lower()

    if file_extension == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif file_extension == ".docx":
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    else:
        return uploaded_file.read().decode("utf-8")


@st.cache_resource
def load_embedder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')





@st.cache_resource
def data_base():

    db_path = Path("database/chroma")

    db_path.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(path=str(db_path))
  
    try:
        collection_new = client.get_collection(name="company_docs_v2")
    except Exception:
   
        collection_new = client.create_collection(name="company_docs_v2")
       
    return collection_new


def inspect_db(collection_new):
    data = collection_new.get()
    total_docs = len(data["ids"])
    rows =[]
    for i in range(total_docs):
        doc_id = data["ids"][i]
        document = data["documents"][i]
        preview = ( document[:80] + "..." if len(document)>80 else document)
        characters = len(document)
        rows.append({ "ID" : doc_id,
                  "Preview" : preview,
                  "Characters" : characters
                  })
    df = pd.DataFrame(rows)
    return df
                
            

def build_metadata():



def add_to_db(text,collection_new,embedder):
    print("🔴 تابع add_to_db اجرا شد!")
    chunks = chunking(text)
    embeddings = embedder.encode(chunks)

    current_docs_count = collection_new.count()
    for i,(chunk,emb) in enumerate(zip(chunks,embeddings)):
        collection_new.add(
            ids=[f"doc_{current_docs_count+i}"],
            embeddings = [emb.tolist()],
            documents=[chunk])
    return len(chunks)

def search(query, embedder, collection_new, top_k=7):
    
    query_embedding = embedder.encode([query])
    results = collection_new.query(
        query_embeddings = query_embedding.tolist(),
        n_results=top_k)

    return results['documents'][0]
    


def build_conversation_context(messages, max_history=7):
    
    if len(messages) <= 1:
        return ""
    
    context = "Previous conversation:\n"
    recent = messages[-max_history*2:]
    
    for i in range(0, len(recent)-1, 2):
        if i+1 < len(recent):
            user_msg = recent[i]["content"]
            assistant_msg = recent[i+1]["content"]
            context += f"- User asked: \"{user_msg}\"\n"
            context += f"- Assistant answered: \"{assistant_msg[:100]}...\"\n"
    
    return context
