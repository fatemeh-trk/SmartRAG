import streamlit as st
from functions import (
    chunking,
    read_uploaded_file,
    load_embedder,
    data_base,
    add_to_db,
    search
)
from streamlit_c import run_ui

def main():
   
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    
    embedder = load_embedder()
    collection_new = data_base()
    
    
    run_ui(collection_new, embedder)

if __name__ == "__main__":
    main()
