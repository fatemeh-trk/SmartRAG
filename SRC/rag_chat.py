import streamlit as st
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="TechHive ",page_icon="🤖")
st.title("TechHive (RAG version)")
st.caption("smart search in company documents")


def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
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

@st.cache_resource
def data_base():
   
    with open("data/company_info.txt",'r',encoding='utf-8')as f:
        
        com_text = f.read()

    chunks = chunking(com_text)


    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedder.encode(chunks)


    client = chromadb.Client()
    try:
        collection_new = client.get_collection(name="company_docs_v2")
    except:
   
        collection_new = client.create_collection(name="company_docs_v2")
    

    for i,(chunk,emb) in enumerate(zip(chunks,embeddings)):
        collection_new.add(
            ids=[f"doc_{i}"],
            embeddings = [emb.tolist()],
            documents=[chunk])
       
    return collection_new, embedder


def search(query, embedder, collection_new, top_k=7):
    
    query_embedding = embedder.encode([query])
    results = collection_new.query(
        query_embeddings = query_embedding.tolist(),
        n_results=top_k)

    return results['documents'][0]
    


with st.spinner("🔄 در حال آماده‌سازی دیتابیس هوشمند..."):
    collection_new, embedder = data_base()

st.sidebar.success("✅ دیتابیس هوشمند آماده است")  
    

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("سوال خود را بپرسید...")

if prompt:
   
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
  
    with st.chat_message("assistant"):
        with st.spinner("🔍 جستجو در اطلاعات شرکت..."):
            
           
            relevant_contexts = search(prompt, embedder, collection_new, top_k=7)
            
            
            context = "\n\n---\n\n".join(relevant_contexts)
            
            
            with st.expander("📚 متن‌های مرتبط پیدا شده"):
                st.caption(context)
            
            
            response = ollama.chat(
                model='phi3',
                messages=[
                    {
                        'role': 'system',
                        'content': f"""You are a helpful assistant for DaneshAfzar company.

IMPORTANT RULES:
1. Answer ONLY based on the document below.
2. If the answer is not in the document, say "I don't have that information."
3. Answer in ENGLISH, short and direct.
4. Use EXACT information from the document (dates, names, numbers).

DOCUMENT:
{context}

YOUR ANSWER (English, short, only from document):"""
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.0,
                    'num_predict': 128,
                }
            )
            
            st.markdown(response['message']['content'])
    
    st.session_state.messages.append({"role": "assistant", "content": response['message']['content']})    
