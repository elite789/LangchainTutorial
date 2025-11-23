import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load Database dari Folder (Bukan dari PDF lagi!)
print("cpu Memuat 'Otak' dari Disk...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma(
    persist_directory="./chroma_db", # Baca dari folder yang dibuat ingest.py
    embedding_function=embeddings,
    collection_name="knowledge_base_perusahaan"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup LLM & Chain (Sama seperti sebelumnya)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

system_prompt = (
    "Anda adalah Asisten. Jawab berdasarkan konteks berikut:\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 3. Loop Chat Interaktif
print("🤖 Bot Siap! (Ketik 'exit' untuk keluar)")
while True:
    query = input("\nKamu: ")
    if query.lower() == "exit":
        break
    
    response = rag_chain.invoke({"input": query})
    print(f"Bot: {response['answer']}")