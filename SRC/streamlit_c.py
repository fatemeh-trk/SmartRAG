import streamlit as st
from functions import chunking , read_uploaded_file , load_embedder,data_base , add_to_db,search
import ollama
import chromadb
from functions import search, add_to_db, read_uploaded_file,build_conversation_context,inspect_db
import time


def run_ui(collection_new, embedder):


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

        if show_debug:
            df = inspect_db(collection_new)
            st.dataframe(df, use_container_width=True)


        
        st.header("📂 افزودن دانش جدید")
        st.sidebar.info(f"🧠 ربات آخرین 7 سوال را به خاطر می‌آورد")
        uploaded_file = st.file_uploader(
            "فایل خود را آپلود کنید (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="file_uploader_unique"
        )
        
        
    
        if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
            with st.status("📥 در حال پردازش فایل..."):
                file_text = read_uploaded_file(uploaded_file)
                num_chunks = add_to_db(file_text, collection_new, embedder)
                
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
            
           
                relevant_contexts = search(prompt, embedder, collection_new, top_k=7)
            
            
                context = "\n\n---\n\n".join(relevant_contexts)
            

                # ساخت زمینه مکالمه (بدون سوال فعلی)
                conv_context = build_conversation_context(st.session_state.messages[:-1])

                with st.expander("📚 متن‌های مرتبط پیدا شده"):
                    st.caption(context)

                response = ollama.chat(
                    model='phi3',
                    messages=[
                    {
                        'role': 'system',
                        'content': f"""You are a helpful assistant.

                        ## CONVERSATION HISTORY:
                        {conv_context}

                        ## INFORMATION FROM UPLOADED FILES:
                        {context}

                        ## CURRENT QUESTION:
                        {prompt}

## RULES:
1. Answer based on the uploaded files first.
2. If information not in files, use conversation history.
3. If not in either, say "I don't have that information."
4. Answer in ENGLISH, short and direct.
5. Do NOT repeat previous answers."""
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
                    
            st.session_state.messages.append({"role": "assistant", "content": answer})
