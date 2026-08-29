def build_context(results):
    document = results["documents"]
    if document and document[0] :
        
            docs = document[0]
            metadata = results["metadatas"][0]
            

            zipped_context = zip(docs , metadata)
            context_parts = []
            sources = []
            for doc_ , meta_data in zipped_context:
        
                file_name = meta_data["file_name"]
                chunk_index = meta_data["chunk_index"]
                sources.append({"chunk_index": chunk_index,
                               "file_name":file_name
                                })
                formatted_chunk = f"""[chunk {chunk_index}]
                            Source: {file_name}
                            
                            {doc_}"""
                context_parts.append(formatted_chunk)
            context = "\n-----------------------------\n".join(context_parts)

        
            return context,sources

     
    else:
        return "",[]

def is_context_empty(context):
    return context.strip() == ""
