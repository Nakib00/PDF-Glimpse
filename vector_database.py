from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# upload PDF and load it
pdf_directory = "pdfs/"

def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def extract_text_with_ocr(file_path):
    """
    Extracts text from a PDF file using OCR.
    This function is suitable for scanned PDFs or PDFs with images containing text.
    """
    # IMPORTANT: Update this path to where you extracted Poppler
    poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin" 
    
    images = convert_from_path(file_path, poppler_path=poppler_path)
    text = ""
    for image in images:
        # You may also need to tell pytesseract where Tesseract-OCR is installed
        # Update this path if Tesseract is not in your system's PATH
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text += pytesseract.image_to_string(image)
    return text

def load_pdf(file_path):
    """
    Loads a PDF and extracts text. It first tries PDFPlumberLoader for text-based PDFs,
    and if that fails to extract substantial text, it falls back to OCR.
    """
    # Try with PDFPlumberLoader first
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    # Check if text was extracted
    if documents and documents[0].page_content.strip():
        # If text is found, return the documents
        return documents
    else:
        # If no text is found, use OCR
        ocr_text = extract_text_with_ocr(file_path)
        # Create a single document from the OCR text
        from langchain_core.documents import Document
        return [Document(page_content=ocr_text)]


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