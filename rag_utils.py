import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_doc_id(content:str) -> str:

    return hashlib.md5(content.encode('utf-8')).hexdigest()

def clean_and_split_documents(docs:list[Document],chunk_size=512,chunk_overlap=50) -> list[Document]:

    for doc in docs:
        doc.metadata["source_system"]="internal_kb"
        if "author" not in doc.metadata:
            doc.metadata["author"]="unknown"

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""] 
    )
    return splitter.split_documents(docs)
