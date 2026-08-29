import pandas as pd
from ingestion.chunker import chunking
from storage.metadata import metadata_builder

def get_files_info(collection_new):
    data = collection_new.get()
    metadatas = data["metadatas"]
    files = {}
    for metadata in metadatas:
        file_id = metadata["file_id"]
        file_name = metadata["file_name"]
        if file_id not in files:
            files[file_id]= {
                "file_name" : file_name,
                "upload_time" :metadata["upload_time"],
                "chunk_count" : 1}
        else:
            files[file_id]["chunk_count"] += 1
    return files   


def files_info_to_dataframe(files):
    rows =[]
    for file_id , file_info in files.items():
        rows.append({
            "ID" : file_id,
            "File name" : file_info["file_name"],
            "Chunk count": file_info["chunk_count"],
            "Upload Time": file_info["upload_time"],
            
            })
        
    df = pd.DataFrame(rows)
    return df


def delete_file(collection_new,file_id):
    try:
        collection_new.delete(where={"file_id":file_id})
        return True


    except Exception as e:
        return False


def get_file_chunks(collection_new,file_id):
    try:
        chunk_file = collection_new.get(
            where={
                "file_id":file_id})
        return chunk_file

    except Exception as e:
        return None


def chunk_transformation(chunk_file):
    chunks = []
    ids = chunk_file["ids"]
    txt = chunk_file["documents"]
    metadata = chunk_file["metadatas"]
    ziped_chunk_data = zip(ids,txt,metadata)
    for chunk_id , doc , data in ziped_chunk_data:
        chunks.append({
            "chunk_id" : chunk_id ,
            "chunk_index" : data["chunk_index"],
            "text" : doc,
            "text_length" : len(doc)
            })
    return chunks    
        
def chunks_to_df(chunks):
    mapped_chunk = {}
    rows = []
    for chunk in chunks:
        mapped_chunk[chunk["chunk_index"]] = chunk["chunk_id"]
        
        rows.append({
            "chunk index" : chunk["chunk_index"],
            "preview" : chunk["text"][:70],
            "text length" : chunk["text_length"]
            
             })
    df = pd.DataFrame(rows)
    return df,mapped_chunk

def delete_selected_chunks(collection_new,ids_to_delete):
    try:
        if ids_to_delete:
            collection_new.delete(ids = ids_to_delete)
            
        return len(ids_to_delete)

    except Exception as e:
        return None

def add_to_db(text,collection_new,embedder,file_name):
    print("🔴 تابع add_to_db اجرا شد!")
    chunks = chunking(text)
    embeddings = embedder.encode(chunks)
    metadatas = metadata_builder(file_name, len(chunks))
    current_docs_count = collection_new.count()
    for i,(chunk,emb) in enumerate(zip(chunks,embeddings)):
        collection_new.add(
            ids=[f"doc_{current_docs_count+i}"],
            embeddings = [emb.tolist()],
            documents=[chunk],
            metadatas = [metadatas[i]]
            )
    
    return len(chunks)
