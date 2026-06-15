import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# USING THE REALTIVE PATHS FOR GETTING THE DATA FROM THE ROOT STRUCTURE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(BASE_DIR, "sample_data")
CHROMA_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")
EMBEDDING_MODEL = "mxbai-embed-large"

# PROCESSING THE INPUT DATA TO FEED THE BOT

def process_doc():
    """Scans the data from the data directory for PDFs or CSVs, chunks them, and generates embeddings"""

    # LOCATE ALL THE PDFs from the targeted directory

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    # ERROR FINDING IN THE PDFs

    if not pdf_files:
        print(f"Error: No PDF files ditected in '{DATA_DIR}'.")
        print("Please place your financial sheets int the data directory")
        return
    
    print(f"Found {len(pdf_files)} PDF(s). Starting data integraion...")

    # EXTRACT TEXT FROM THE PDF(s) IN THE DATA FOLDER

    raw_documents = []
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        print(f"-> Loading and Tagging: {filename}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = filename
        
        raw_documents.extend(docs)

    print(f"Sucessfully extracted {len(raw_documents)} total pages.")

    # CHUNK THE TEXT INTO PIECES
    # 1000 TOKENS WITH 200 TOKEN OVERLAP ENSURES PARAGRAPHS AND FINANCIAL VALUES

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        is_separator_regex = False
    )

    document_chunks = text_splitters.split_documents(raw_documents)
    print(f"Split documents into {len(document_chunks)} chunks.")

    # INITIALIZE THE LOCAL OLLAMA EMBEDDING MODEL

    print(f"Initializing the local embedding model ({EMBEDDING_MODEL})")
    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)

    # GENERATE VECTORS AND PERIST TO CHROMADB

    print("Generating mathematial vectors...")

    # CHROMA DETECTS AND SAVE IN THE LOCAL DRIVE IN THE DATABASE

    db = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Ingestion complete Vector dataase sucessfull save to '{CHROMA_PATH}'.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)

    process_doc()