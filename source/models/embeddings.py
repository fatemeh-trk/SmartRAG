import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2'
    )


