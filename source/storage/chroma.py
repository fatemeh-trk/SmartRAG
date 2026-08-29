from pathlib import Path
import chromadb
import pandas as pd

def data_base():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    db_path = PROJECT_ROOT / "storage" / "chroma"

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
        metadata = data["metadatas"][i]
        rows.append({ "ID" : doc_id,
                  "Preview" : preview,
                  "Characters" : characters,
                  "File" : metadata["file_name"],
                  "Chunk": metadata["chunk_index"],
                  "Upload Time": metadata["upload_time"],
                  "File ID": metadata["file_id"]

                  })
    df = pd.DataFrame(rows)
    return df
