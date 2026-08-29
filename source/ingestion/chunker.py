from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
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
