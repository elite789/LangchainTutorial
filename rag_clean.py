import os
from dotenv import load_dotenv

# 1. Import Komponen Modern
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 2. Import Chain & Prompt (Bagian Paling Penting)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setup Environment
load_dotenv()
# Pastikan key ada
if not os.getenv("GOOGLE_API_KEY"):
    print("⚠️ GOOGLE_API_KEY belum di-set!")

# --- TAHAP 1: PERSIAPAN DATA (INGESTION) ---
print("📂 1. Memuat & Memecah Dokumen...")
# Ganti path sesuai file Anda
loader = PyPDFLoader("/Users/azhardzakwan/Documents/To_DriveAzhar/Coding/AgenticAI/LangChain/PDF Langchain Test.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# --- TAHAP 2: OTAK & MEMORI (LLM & VECTOR DB) ---
print("🧠 2. Membuat Embeddings & Vector Store...")
# Menggunakan model embedding Google
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Membuat Vector DB (Chroma)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="clean_rag_collection"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Menyiapkan LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# --- TAHAP 3: MERAKIT RAG (THE CHAIN) ---
print("🔗 3. Merakit Rantai RAG Modern...")

# A. Membuat Prompt (Instruksi)
# Perhatikan struktur ("system", "human") ini standar untuk Chat Model
system_prompt = (
    "Anda adalah Asisten HRD yang tegas. "
    "Jawab pertanyaan user berdasarkan konteks berikut:\n\n"
    "{context}"
    "\n\nJika tidak ada di konteks, katakan: 'Maaf, info tidak ada di SOP'."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# B. Chain 1: Dokumen + LLM (Question Answer Chain)
# Chain ini tugasnya: "Ambil teks dokumen, masukkan ke prompt, kirim ke LLM"
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# C. Chain 2: Retriever + Chain 1 (RAG Chain Utama)
# Chain ini tugasnya: "Cari dokumen dulu (Retriever), lalu oper ke Chain 1"
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- TAHAP 4: EKSEKUSI ---
def tanya_hrd(pertanyaan):
    print(f"\n❓ User: {pertanyaan}")
    
    # 'input' adalah variabel yang kita definisikan di prompt ("human", "{input}")
    response = rag_chain.invoke({"input": pertanyaan})
    
    print(f"💡 Bot: {response['answer']}")
    
    # Debugging: Lihat dokumen mana yang dibaca bot
    print("\n   [Sumber Data]:")
    for i, doc in enumerate(response['context']):
        print(f"   - Hal {doc.metadata.get('page', '?')}: {doc.page_content[:50]}...")

# Test
tanya_hrd("Bagaimana prosedur cuti?") 
# Harusnya menjawab tidak tahu (karena tidak ada di PDF)