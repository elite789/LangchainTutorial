import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import Library RAG (LangChain)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setup Aplikasi Flask
app = Flask(__name__)
load_dotenv()

# --- 1. MEMUAT OTAK AI (RAG) ---
print("⚙️  Sedang memuat Database Knowledge Base...")

# Cek apakah folder database ada
if not os.path.exists("./chroma_db"):
    print("❌ ERROR: Folder 'chroma_db' tidak ditemukan! Harap jalankan 'ingest.py' atau 'ingest_universal.py' terlebih dahulu.")
    rag_chain = None
else:
    # Setup Embeddings & DB
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="knowledge_base_perusahaan"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Setup LLM & Prompt
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    
    system_prompt = (
        "Anda adalah Asisten Bot Internal. Jawab pertanyaan berdasarkan konteks berikut. "
        "Jika informasi tidak ada di dokumen, katakan: 'Maaf, informasi tidak ditemukan di database'.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Bot Siap! Menunggu pesan dari Google Chat...")


# --- 2. SETUP JALUR KOMUNIKASI (ENDPOINT) ---
@app.route('/', methods=['POST'])
def on_event():
    """Fungsi ini dipanggil otomatis oleh Google Chat setiap ada pesan"""
    event = request.get_json()
    
    # Skenario A: Bot baru diundang ke Space/DM
    if event['type'] == 'ADDED_TO_SPACE':
        return jsonify({'text': 'Halo! Saya Asisten Dokumen. Silakan tanya saya tentang SOP/Data.'})
    
    # Skenario B: Pesan Masuk (MESSAGE)
    if event['type'] == 'MESSAGE':
        # Ambil teks pesan user
        user_message = event['message']['text']
        # Bersihkan nama bot (jika di-mention di grup)
        clean_message = user_message.replace(event['message'].get('argumentText', ''), '').strip() or user_message
        
        print(f"📩 Pesan Masuk: {clean_message}")

        if not rag_chain:
            return jsonify({'text': '⚠️ Error: Database belum siap. Cek server.'})

        # Tanya ke RAG
        try:
            response = rag_chain.invoke({"input": clean_message})
            answer = response['answer']
            
            # (Opsional) Tampilkan sumber halaman
            sources = []
            for doc in response['context']:
                page = doc.metadata.get('page', '?')
                source_file = os.path.basename(doc.metadata.get('source', 'Doc'))
                sources.append(f"{source_file} (Hal {page})")
            
            # Format Balasan
            source_text = f"\n\n📚 *Sumber:* {', '.join(sources)}" if sources else ""
            final_reply = f"{answer}{source_text}"
            
            return jsonify({'text': final_reply})
            
        except Exception as e:
            print(f"Error RAG: {e}")
            return jsonify({'text': f"Maaf, terjadi kesalahan sistem: {str(e)}"})

    return jsonify({})

if __name__ == '__main__':
    # Jalankan server di port 5000
    app.run(port=5000)