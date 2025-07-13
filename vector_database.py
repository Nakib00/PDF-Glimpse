from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# upload PDF and load it
pdf_directory = "pdfs/"

def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Create chunks from the loaded PDF
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    text_chunks = text_splitter.split_documents(documents)

    return text_chunks

# enmbedding model deepseek using ollama
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

# store the chunks in a vector database
FAISS_db_path = "vector_db/db_faiss"

def create_vector_db(text_chunks, ollama_model_name):
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
    faiss_db.save_local(FAISS_db_path)
    return faiss_db