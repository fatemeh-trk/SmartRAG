import uuid
from datetime import datetime

def metadata_builder(file_name,num_chunks):
    file_id = str(uuid.uuid4())
    metadata_list = []
    upload_time = datetime.now().isoformat()
    
    for i in range(num_chunks):
        chunk_index = i
        metadata_list.append({
                 "file_id" : file_id,
                 "upload_time" : upload_time,
                 "file_name" :file_name,
                 "chunk_index": chunk_index
                 
            })

    return metadata_list
        
