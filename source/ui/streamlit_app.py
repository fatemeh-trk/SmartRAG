import streamlit as st
import ollama
import time


from ingestion.loaders import read_uploaded_file



from storage.chroma import data_base, inspect_db

from services.document_service import (
    get_files_info,
    files_info_to_dataframe,
    get_file_chunks,
    chunks_to_df,
    chunk_transformation,
    add_to_db,
    delete_file,
    delete_selected_chunks,
)

from retrieval.search import search, process_retrieval_results
from retrieval.reranker import rerank

from generation.context_builder import build_context, is_context_empty
from generation.prompts import get_system_prompt

def group_sources(sources):
    grouped_src ={}
    
    for src in sources:
       file_name = src["file_name"]
       chunk_index = src["chunk_index"]
       
       if file_name not in grouped_src:

            grouped_src[file_name] = []
            
       grouped_src[file_name].append(chunk_index)

    return grouped_src




def run_ui(collection_new, embedder,reranker):

    
        
        
    st.set_page_config(page_title="TechHive ",page_icon="🤖")
    st.title("TechHive (RAG version)")
    st.caption("smart search in company documents")



    st.sidebar.success("✅ دیتابیس هوشمند آماده است")
    

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    with st.sidebar:

        show_debug = st.checkbox("🔍 نمایش Database Inspector")
        if "success_message" in st.session_state:
            st.success(st.session_state["success_message"])
            del st.session_state["success_message"]
        if show_debug:
            
            
            #df = inspect_db(collection_new)
            #st.dataframe(df, use_container_width=True)
            files = get_files_info(collection_new)
            df = files_info_to_dataframe(files)
            st.dataframe(df, width="stretch")
            file_names = []
            for file_id , file_info in files.items():
                file_names.append(file_info["file_name"])
                
            selected_file = st.radio(
                "📄 فایل مورد نظر را انتخاب کنید:",
                file_names,
                index=None
            )

            selected_file_id = None
            for file_id,file_info in files.items():
                if file_info["file_name"] == selected_file :
                    selected_file_id = file_id
            if selected_file_id != None :
                chunk_file = get_file_chunks(collection_new,selected_file_id)
                tran_chunk = chunk_transformation(chunk_file)
                chunk_df,mapped_chunk = chunks_to_df(tran_chunk)
                chunk_df["select"] = False
                edited_chunk_df = st.data_editor(chunk_df)
                selected_chunks = edited_chunk_df[
                    edited_chunk_df["select"]== True]
                ids_to_delete = []
                for index,row in selected_chunks.iterrows():
                    chunk_index = row["chunk index"]
                    chunk_id = mapped_chunk[chunk_index]
                    ids_to_delete.append(chunk_id)
                chunk_deletion_result = None
                if st.button("حذف چانک"):
                    chunk_deletion_result = delete_selected_chunks(collection_new,ids_to_delete)
                    if chunk_deletion_result > 0 :
                
                        st.session_state["success_message"] = f"{chunk_deletion_result}چانک حذف شد "
                        st.rerun()
                        

                        
                    elif chunk_deletion_result == 0:
                        st.warning("چانک انتخاب نشده")
                    else:
                        st.error(" حذف چانک با مشکل موجه شد")
                
            
                    
                    
                
                    
                    
            else:
                st.write("no chunk")
            file_deletion_result = None
            if st.button("حذف فايل"):
                file_deletion_result = delete_file(collection_new , selected_file_id)
                if file_deletion_result :
                
                    st.success("فايل با موفقيت حذف شد")
                    
                    st.rerun()
                
                else:
                    st.error("حذف فايل با مشکل موجه شد")
                
            


        
        st.header("📂 افزودن دانش جدید")
        
        uploaded_file = st.file_uploader(
            "فایل خود را آپلود کنید (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="file_uploader_unique"
        )
        
        
    
        if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
            with st.status("📥 در حال پردازش فایل..."):
                file_text = read_uploaded_file(uploaded_file)
                num_chunks = add_to_db(file_text, collection_new, embedder,uploaded_file.name)
                
                st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"✅ فایل '{uploaded_file.name}' پردازش شد!")
            st.info(f"📊 {num_chunks} قطعه به دانش اضافه شد")
        elif uploaded_file is not None:
            st.info(f"ℹ️ فایل '{uploaded_file.name}' قبلاً پردازش شده است")
        
        st.divider()
    
        st.header("📊 وضعیت دیتابیس")
        total_docs = collection_new.count()
        st.metric("تعداد کل قطعات", total_docs)
 

        st.divider()
        st.header("🗑️ مدیریت دیتابیس")
    

        if total_docs == 0:
            st.info("✨ دیتابیس خالی است. لطفاً فایلی آپلود کنید.")
    
        else:
            delete_method = st.radio(
                "روش حذف:",
                ["❌ حذف به تعداد", "📋 حذف با آیدی", "🗑️ حذف همه"],
                index=None
            )
        
        
            if delete_method == "❌ حذف به تعداد":
            
                default_value = min(5, total_docs)
            
                num_to_delete = st.number_input(
                    "تعداد قطعات برای حذف (از آخرین)",
                    min_value=1,
                    max_value=total_docs,
                    value=default_value
                )
            
                if st.button("🗑️ حذف آخرین قطعات", type="secondary"):
                    all_ids = collection_new.get()['ids']
                    ids_to_delete = all_ids[-num_to_delete:]
                    collection_new.delete(ids=ids_to_delete)
                    st.success(f"✅ {num_to_delete} قطعه حذف شد")
                    time.sleep(4)
                    st.rerun()
        
        
            elif delete_method == "📋 حذف با آیدی":
                all_ids = collection_new.get()['ids']
            
                if len(all_ids) <= 30:
                    selected = st.multiselect("انتخاب قطعات:", all_ids)
                    if selected and st.button("🗑️ حذف", type="secondary"):
                        collection_new.delete(ids=selected)
                        st.success(f"✅ {len(selected)} قطعه حذف شد")
                        time.sleep(4)
                        st.rerun()
                  
                else:
                    st.caption(f"{len(all_ids)} قطعه موجود. آیدی‌ها از doc_0 تا doc_{total_docs-1}")
                    selected_id = st.text_input("آیدی قطعه (مثال: doc_5):")
                    if selected_id and st.button("🗑️ حذف", type="secondary"):
                        collection_new.delete(ids=[selected_id])
                        st.success(f"✅ قطعه {selected_id} حذف شد")
                        time.sleep(4)
                        st.rerun()
        
        
            elif delete_method == "🗑️ حذف همه":
                st.warning("⚠️ این عمل همه قطعه‌ها را حذف می‌کند!")
                if st.checkbox("تأیید حذف همه قطعه‌ها"):
                    try:
            
                        all_ids = collection_new.get()['ids']
                        if all_ids:
                            collection_new.delete(ids=all_ids)
                            st.success(f"✅ {len(all_ids)} قطعه با موفقیت حذف شدند")
                        else:
                            st.info("دیتابیس قبلاً خالی است")
                            
                        time.sleep(4)
                        st.rerun()
                    except Exception as e:
                        st.error(f"خطا در حذف: {e}")
                
                
                                  
        st.header("💡 راهنما")
        st.markdown("""
-اين ربات در حال حاضر فقط از زبان انگليسي حمايت ميکند
    - فایل‌های PDF، Word و TXT پشتیبانی می‌شوند
    - هرچه فایل بزرگتر باشد، پردازش بیشتر طول می‌کشد
    - سوالات خود را به انگلیسی یا فارسی بپرسید
    """)   


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
            
           
                results = search(prompt, embedder, collection_new, top_k=20)
                
                
                NO_INFORMATION_MESSAGE = "I don't have that information."

# RETRIEVAL_THRESHOLD tuned empirically via retrieval_evaluation.csv
# Chroma default distance metric = L2 (squared euclidean), unnormalized embeddings
# from 'paraphrase-multilingual-MiniLM-L12-v2'.
# Observed: relevant matches cluster ~7-20, irrelevant/negative queries cluster ~24-34.
# Re-validate this threshold if the embedding model changes.
                RETRIEVAL_THRESHOLD = 21

                
                results = process_retrieval_results(results, RETRIEVAL_THRESHOLD)
                results = rerank(
                    prompt,
                    results,
                    reranker,
                    top_n=7
                        )
                
                
                print("AFTER RERANK COUNT:", len(results["documents"][0]))

                for i, meta in enumerate(results["metadatas"][0]):
                    print(
                        "FINAL:",
                         i,
                         meta["file_name"],
                         meta["chunk_index"]
                            )
            
                context,sources = build_context(results)
            
                grouped = group_sources(sources)
                
                

                
                if not is_context_empty(context):
                    with st.expander("📚 متن‌های مرتبط پیدا شده"):
                        st.caption(context)
                    response = ollama.chat(
                        model='phi3',
                        messages=[
                        {
                            'role': 'system',
                            'content': get_system_prompt(context)
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                    ],
                    options={
                        'temperature': 0.0,
                        'num_predict': 128,
                    }
                    )
                    answer = response['message']['content']
                    st.markdown(answer)
                    st.markdown("### 📄  Sources")
                    for file_name , chunks in grouped.items():
                        st.markdown(file_name)
                        chunk_str = ", ".join([str(c) for c in sorted(set(chunks))])
                    
                        st.markdown(f"chunk(s):{chunk_str}")
                else:
                    st.markdown(NO_INFORMATION_MESSAGE)
                    
            
