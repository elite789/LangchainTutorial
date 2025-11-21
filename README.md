# LangChain Tutorial

A comprehensive tutorial project demonstrating PDF processing, embeddings, and RAG (Retrieval Augmented Generation) using LangChain with Google Gemini.

## 📋 Features

- **PDF Document Loading**: Load and parse PDF documents using PyPDFLoader
- **Text Chunking**: Split documents into manageable chunks with RecursiveCharacterTextSplitter
- **Vector Embeddings**: 
  - Google Gemini embeddings (API-based)
  - HuggingFace embeddings (local, no API required)
- **Vector Storage**: Store embeddings in ChromaDB for efficient retrieval
- **RAG Implementation**: Question-answering system using retrieval-augmented generation
- **Multiple LLM Support**: Integration with Google Gemini models

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- Google API Key (for Gemini models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/elite789/LangchainTutorial.git
cd LangchainTutorial
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## 📁 Project Structure

```
.
├── pdf_chunk.py           # Main script using Google Gemini embeddings
├── pdf_chunk_local.py     # Alternative script using local HuggingFace embeddings
├── langchain_example.py   # Basic LangChain examples
├── PDF Langchain Test.pdf # Sample PDF document
├── .env                   # Environment variables (not tracked in git)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 💻 Usage

### Using Google Gemini Embeddings

```bash
python pdf_chunk.py
```

This script:
- Loads a PDF document
- Splits it into chunks
- Creates embeddings using Google Gemini
- Stores vectors in ChromaDB
- Implements a RAG-based Q&A system

### Using Local HuggingFace Embeddings (No API Required)

```bash
python pdf_chunk_local.py
```

This alternative script:
- Uses local HuggingFace embeddings (all-MiniLM-L6-v2)
- No API quota limits
- Runs entirely offline after initial model download

## 🔑 API Keys

This project requires a Google API key for Gemini models. Get your free API key at:
- [Google AI Studio](https://makersuite.google.com/app/apikey)

**⚠️ Security Note**: Never commit your `.env` file or expose your API keys!

## 📦 Dependencies

Main packages:
- `langchain` - LangChain framework
- `langchain-community` - Community integrations
- `langchain-google-genai` - Google Gemini integration
- `langchain-classic` - Legacy chains support
- `chromadb` - Vector database
- `pypdf` - PDF processing
- `sentence-transformers` - Local embeddings (optional)
- `python-dotenv` - Environment variable management

## 🛠️ Key Components

### Document Loading
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()
```

### Text Splitting
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### Vector Storage
```python
from langchain_community.vectorstores import Chroma
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection"
)
```

### RAG Chain
```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "Your question here"})
```

## 🐛 Troubleshooting

### Quota Exceeded Error
If you encounter Google API quota errors, use `pdf_chunk_local.py` which uses local embeddings.

### Module Not Found Errors
Make sure you've installed all dependencies:
```bash
pip install langchain langchain-community langchain-google-genai langchain-classic chromadb pypdf python-dotenv
```

### Pip Version Issues
If you encounter pip errors, downgrade to a stable version:
```bash
python -m pip install --upgrade pip==24.3.1
```

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests!

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🎓**
