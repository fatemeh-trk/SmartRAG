import streamlit as st

from models.embeddings import load_embedder
from models.reranker import load_reranker
from storage.chroma import data_base
from ui.streamlit_app import run_ui


def main():
   
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    
    embedder = load_embedder()
    reranker = load_reranker()
    collection_new = data_base()
    
    
    run_ui(collection_new, embedder,reranker)

if __name__ == "__main__":
    main()
