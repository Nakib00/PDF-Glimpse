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

# test the PDF loading functionality
file_path = 'Universal_Declaration_of_Human_Rights.pdf'
documents = load_pdf(file_path)
# print(f"Loaded {len(documents)} documents from {file_path}")


# Create chunks from the loaded PDF
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks

text_chunks = create_chunks(documents)
# print(f"Created {len(text_chunks)} text chunks from the PDF.")

# enmbedding model deepseek using ollama
ollama_model_name = "deepseek-r1:1.5b"

def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name) 
    return embeddings

# store the chunks in a vector database
FAISS_db_path = "vector_db/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_db_path)