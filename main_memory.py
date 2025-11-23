import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

#Loading Database
print("Loading Data from Vector DB...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="/Users/azhardzakwan/Documents/To_DriveAzhar/Coding/AgenticAI/LangChain/chroma_db",
    embedding_function=embeddings,
    collection_name="knowledge_base_perusahaan"
)
retriever = vectorstore.as_retriever(search_kwargs={"k" : 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# --- MEMBUAT SUB-CHAIN: REFORMULASI PERTANYAAN ---
# Tugas: Mengubah pertanyaan ambigu ("Kalau itu?") menjadi pertanyaan lengkap.

contextualize_q_system_prompt = (
    "Diberikan riwayat percakapan dan pertanyaan pengguna terbaru "
    "yang mungkin merujuk pada konteks sebelumnya, "
    "rumuskan kembali menjadi pertanyaan mandiri yang dapat dipahami "
    "tanpa melihat riwayat percakapan. "
    "JANGAN dijawab pertanyaannya, cukup rumuskan ulang saja jika perlu."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),  #Chat history
        ("human", "{input}")
    ]
)

#Re-Formatting question sebelum di pass kembali ke retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# --- MEMBUAT SUB-CHAIN: JAWAB PERTANYAAN (QA) ---""
# Tugas: Menjawab pertanyaan yang sudah diperbaiki menggunakan dokumen.

qa_system_prompt = (
    "Anda adalah asisten tanya jawab tugas. "
    "Gunakan potongan konteks berikut untuk menjawab pertanyaan. "
    "Jika tidak tahu, katakan tidak tahu. "
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- MANAJEMEN SESSION (PENYIMPANAN MEMORI) ---
# Kita butuh tempat untuk menyimpan history per user (Session ID)

store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#Combine RAG dengan Memory

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "chat_history",
    output_messages_key = "answer"
)

print("🤖 Bot Siap Ngobrol! (Ketik 'exit' untuk keluar)")
session_id = "user_budi_123" # Anggap ini ID unik user

while True:
    user_input = input("\nKamu: ")
    if user_input.lower() == "exit":
        break

    response = conversational_rag_chain.invoke(
        {"input" : user_input},
        config={"configurable" : {"session_id" : session_id} } 
    )

    print(f"Bot: {response['answer']}")
