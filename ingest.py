import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

#1. Loading Data
print("Reading PDF Data ...")
loader = PyPDFLoader("./PDF Langchain Test.pdf")
docs = loader.load()

#2. Split/Chunking Data
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"Data dipecah menjadi {len(splits)} bagian")

#3. Embedding
print("💾 Sedang menyimpan ke Hard Disk (Vector DB)...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="knowledge_base_perusahaan",
    persist_directory="./chroma_db"  # <--- Data disimpan di folder ini
)

print("Database tersimpan di folder './chroma_db'.")