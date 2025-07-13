# PDF-Glimpse

PDF-Glimpse is a powerful and intuitive application that allows you to have a conversation with your PDF documents. Simply upload a PDF, ask a question, and get intelligent, context-aware answers. This tool leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to understand your documents and provide precise information.

---
![PDF Glimpse Web View](https://github.com/Nakib00/PDF-Glimpse/blob/main/view/PDF%20Glimpse%20view.png?raw=true)


## 🚀 How it Works

The application follows a simple yet powerful workflow:

1. **Upload a PDF**: You start by uploading a PDF document through the user-friendly interface.
2. **Ask a Question**: Type your question about the PDF content in the text area.
3. **Document Processing**:
   - The PDF is loaded, and its text is extracted. If the PDF is image-based or scanned, Optical Character Recognition (OCR) is automatically used to extract the text.
   - The text is broken down into smaller, manageable chunks.
   - These chunks are converted into numerical representations (embeddings) and stored in a vector database (FAISS).
4. **Information Retrieval**: When you ask a question, the application searches the vector database to find the most relevant text chunks from the PDF.
5. **Answer Generation**: The retrieved text chunks and your original question are sent to a Large Language Model (LLM), which generates a coherent and accurate answer based *only* on the provided information.

---

## 🛠️ Technologies Used

This project is built with a modern stack of AI and web technologies:

### **Backend & Logic**

- **LangChain**: The core framework for building the RAG pipeline, connecting different components like data loaders, text splitters, and models.
- **Groq**: Provides access to a fast Large Language Model for generating answers.
- **Ollama**: Used to run the text embedding models locally, which convert text into vectors.
- **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search, used here as the vector store for the document embeddings.
- **PDFPlumber**: A robust library for extracting text from PDF files.
- **Pytesseract & PDF2Image**: Used for the OCR (Optical Character Recognition) functionality to extract text from scanned PDFs.

### **Frontend**

- **Streamlit**: An open-source app framework for creating beautiful, custom web apps for machine learning and data science projects in pure Python.

---

## ⚙️ How to Run the Project Locally

Follow these steps to get PDF-Glimpse running on your local machine.

### Prerequisites

- Python 3.8 or higher
- An API key from [Groq](https://console.groq.com/keys)
- **Tesseract-OCR**: This is required for the OCR functionality.
  - **Windows**: Download and install from the [official Tesseract repository](https://github.com/tesseract-ocr/tesseract).
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`
- **Poppler**: This is required to convert PDF pages to images for OCR.
  - **Windows**: Download the latest version from [the official site](https://poppler.freedesktop.org/)
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`

### 1. Clone the Repository

```bash
git clone <repository_url>
cd pdf-glimpse
```
### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv .venv
source .venv\Scripts\activate
```
### 3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```
### 4. Set Up Environment Variables
Create a file named .env in the root directory of the project and add your Groq API key to it. The project uses python-dotenv to automatically load this key.

```bash
GROQ_API_KEY="your_groq_api_key_here"
```

### 6. Configure OCR (Windows Only)
If you are on Windows, you will need to update the paths for Poppler and Tesseract in the vector_database.py file:

```bash
# IMPORTANT: Update this path to where you extracted Poppler
poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"

# Update this path if Tesseract is not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

```

### 7. Run the Application
Start the Streamlit server.

```bash
streamlit run ui.py
```

### How to Use the Application
Using PDF-Glimpse is incredibly simple:

* Upload your PDF file using the file uploader at the top of the page.
* Once the PDF is uploaded, type your question into the text box labeled "Enter your query here:".
* Click the "Ask Question" button.
* The assistant will process the document and display the answer in the chat interface below
