
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document

def read_uploaded_file(uploaded_file):
    file_extension= Path(uploaded_file.name).suffix.lower()

    if file_extension == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif file_extension == ".docx":
        doc = Document(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    else:
        return uploaded_file.read().decode("utf-8")
