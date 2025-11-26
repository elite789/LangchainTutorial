import streamlit as st
import os
import base64 # <-- Library baru untuk encoding PDF
from dotenv import load_dotenv

# Import Library LangChain
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. KONFIGURASI HALAMAN (WIDE MODE) ---
# Penting: layout="wide" agar muat 2 kolom
st.set_page_config(
    page_title="AI Assistant Prototype",
    page_icon="🤖",
    layout="wide" 
)

st.title("🤖 AI Assistant Prototype")

load_dotenv()

# Tentukan Lokasi File PDF Anda (Harus sama dengan yang di ingest.py)
# Gunakan relative path agar aman di device baru
PDF_FILE_PATH = "./PDF Langchain Test.pdf"

# --- FUNGSI TAMPILKAN PDF (KIRI) ---
def display_pdf(file_path):
    # Cek apakah file ada
    if not os.path.exists(file_path):
        st.error(f"File PDF tidak ditemukan di: {file_path}")
        return

    # Baca file dan ubah ke Base64
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embed PDF menggunakan HTML iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- FUNGSI LOAD ENGINE RAG (BACKEND) ---
@st.cache_resource
def get_rag_chain():
    if not os.path.exists("./chroma_db"):
        st.error("Folder 'chroma_db' belum ada. Jalankan ingest.py dulu!")
        return None
        
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="knowledge_base_perusahaan"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Context Prompt
    context_system_prompt = "Rumuskan kembali pertanyaan user menjadi pertanyaan mandiri berdasarkan riwayat chat."
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", context_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # QA Prompt
    qa_system_prompt = (
        "Anda adalah Asisten HRD. Jawab berdasarkan konteks berikut.\n"
        "Jika tidak tahu, katakan 'Maaf, informasi tidak ditemukan di SOP'.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

rag_chain = get_rag_chain()

# --- LAYOUT UTAMA: MEMBAGI LAYAR JADI 2 KOLOM ---
# Ratio [1, 1] artinya lebar kolom sama besar. Bisa diganti [1.5, 1] jika mau PDF lebih lebar.
col1, col2 = st.columns([1, 1]) 

# === KOLOM KIRI (PDF PREVIEW) ===
with col1:
    st.header("📄 Dokumen Sumber")
    display_pdf(PDF_FILE_PATH)

# === KOLOM KANAN (CHAT INTERFACE) ===
with col2:
    st.header("💬 Chat Asisten")
    
    # Init Session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tampilkan History Chat (Hanya di kolom kanan)
    # Kita gunakan container agar chat area punya batas tinggi (opsional)
    chat_container = st.container(height=650) 
    
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

    # Input User
    # Note: st.chat_input secara default menempel di bawah layar, 
    # tapi pesannya akan kita render masuk ke dalam kolom kanan.
    user_input = st.chat_input("Tanya seputar dokumen ini...")

    if user_input:
        # 1. Tampilkan User Input (Di dalam container kolom kanan)
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # 2. Proses Jawaban
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Menganalisa dokumen..."):
                    try:
                        response = rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response['answer']
                        
                        st.markdown(answer)
                        
                        # Expander Referensi
                        with st.expander("🔍 Cek Halaman Sumber"):
                            for doc in response['context']:
                                page = doc.metadata.get('page', '?')
                                st.markdown(f"- **Halaman {page}:** {doc.page_content[:100]}...")
                        
                        st.session_state.chat_history.append(AIMessage(content=answer))
                    except Exception as e:
                        st.error(f"Error: {e}")