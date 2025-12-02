import os
import asyncio
from dotenv import load_dotenv

# Library Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Library RAG (LangChain) - Masih sama persis!
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- KONFIGURASI ---
# Masukkan Token dari BotFather di sini (atau taruh di .env biar aman)
TELEGRAM_TOKEN = os.getenv("bot_tele_token")

# --- 1. SETUP OTAK AI (RAG) ---
print("⚙️  Memuat Database RAG...")

if not os.path.exists("./chroma_db"):
    print("❌ ERROR: Folder 'chroma_db' tidak ditemukan! Jalankan 'ingest_universal.py' dulu.")
    rag_chain = None
else:
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="knowledge_base_perusahaan"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    system_prompt = (
        "Anda adalah Asisten Telegram. Jawab pertanyaan berdasarkan konteks dokumen berikut. "
        "Gunakan bahasa Indonesia yang luwes namun profesional. "
        "Jika tidak ada di dokumen, katakan tidak tahu.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Otak AI Siap!")

# --- 2. FUNGSI TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Respon saat user ketik /start"""
    await update.message.reply_text(
        "Halo! 👋 Saya Asisten Dokumen Pribadi Anda.\n"
        "Silakan kirim pertanyaan, saya akan cari jawabannya di file Drive Anda."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fungsi utama: Terima pesan -> Tanya AI -> Balas"""
    user_text = update.message.text
    user_name = update.effective_user.first_name
    
    print(f"📩 {user_name}: {user_text}")
    
    # Beri status 'Typing...' biar user tahu bot sedang mikir
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    if not rag_chain:
        await update.message.reply_text("⚠️ Error: Database belum siap.")
        return

    try:
        # Panggil RAG Chain (Proses berpikir)
        # Note: Kita jalankan synchronous code di dalam async wrapper
        response = await asyncio.to_thread(rag_chain.invoke, {"input": user_text})
        answer = response['answer']
        
        # (Opsional) Ambil Sumber
        sources = []
        for doc in response['context']:
            page = doc.metadata.get('page', '?')
            source_file = os.path.basename(doc.metadata.get('source', 'Doc'))
            sources.append(f"- {source_file} (Hal {page})")
        
        # Format Balasan
        source_text = "\n\n📚 *Sumber:*\n" + "\n".join(sources) if sources else ""
        final_reply = f"{answer}{source_text}"
        
        # Kirim Balasan (Parse Markdown agar bold/italic jalan)
        await update.message.reply_text(final_reply)

    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text("Maaf, saya pusing. Coba lagi nanti.")

# --- 3. JALANKAN BOT ---
if __name__ == '__main__':
    print("🚀 Bot Telegram Sedang Berjalan...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Daftarkan Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    # Jalankan (Polling Mode - Tidak butuh Ngrok!)
    app.run_polling()