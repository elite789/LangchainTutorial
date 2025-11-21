import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import RetrievalQA
# from langchain.chains import RetrievalQA

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Load API key from environment (if needed for OpenAI)
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

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

print("Processing Embedding (Google)")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#saving to chroma db
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="collection_langchain_pdf_test"
)

print("Data saved as vector to Chroma DB")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True
    )

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# A. Buat Prompt System (Agar AI tahu perannya)
system_prompt = (
    "Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan. "
    "Gunakan potongan konteks berikut untuk menjawab pertanyaan. "
    "Jika Anda tidak tahu jawabannya, katakan bahwa Anda tidak tahu. "
    "Gunakan maksimal tiga kalimat dan jawab dengan ringkas."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# B. Buat Chain untuk Memproses Dokumen (Stuff Documents Chain)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# C. Buat Chain Utama (Retrieval Chain)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, 
#     retriever=retriever, 
#     chain_type = "stuff", 
#     return_source_documents=True)

#Using LCEL(LangChain Expression Language)
def tanya(pertanyaan):
    print(f"\n❓ Tanya: {pertanyaan}")
    # Perhatikan key inputnya adalah "input"
    response = rag_chain.invoke({"input": pertanyaan})
    print(f"💡 Jawab: {response['answer']}")

#Using RetrievalQA
def tanya_bot(pertanyaan):
    print(f"\n. Tanya: {pertanyaan}")
    try:
        response = qa_chain.invoke({"query": pertanyaan})
        print(f"Jawaban: {response['result']}")
        print("Sumber Dokumen : ")
        for i, doc in enumerate(response['source_documents']):
            print(f" [{i+1}] Halaman {doc.metadata.get('page', '?')} : {doc.page_content[:50]}")
    except Exception as e:
        print(f"Error: {e}")

tanya("Siapa CEO Rans Entertainment?")

