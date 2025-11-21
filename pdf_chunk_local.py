import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("/Users/azhardzakwan/Documents/To_DriveAzhar/AgenticAI/LangChain/PDF Langchain Test.pdf")
raw_documents = loader.load()
print(f"Number of pages: {len(raw_documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

documents = text_splitter.split_documents(raw_documents)

print(f"Number of chunks: {len(documents)}")
print(f"First chunk: {documents[0].page_content}")

print("\nProcessing Embedding (HuggingFace - Local, No API needed)")
# Using a lightweight, free local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Small, fast, and effective
    model_kwargs={'device': 'cpu'}
)

print("Creating embeddings and saving to Chroma DB...")
# Saving to chroma db
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="collection_langchain_pdf_test",
    persist_directory="./chroma_db"  # Optional: persist to disk
)

print("✅ Data saved as vectors to Chroma DB successfully!")
print(f"Vector store contains {vector_db._collection.count()} documents")

# Example: Query the vector store
print("\n--- Testing Vector Store Query ---")
query = "What are the working hours?"
results = vector_db.similarity_search(query, k=2)
print(f"\nQuery: '{query}'")
print(f"Top result:\n{results[0].page_content[:200]}...")
