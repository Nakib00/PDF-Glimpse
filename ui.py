from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st
from vector_database import upload_pdf, load_pdf, create_chunks, create_vector_db

st.title("PDF Glimpse")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
user_query = st.text_area("Enter your query here:", height=150, placeholder="Type your question about the PDF...")
ask_question = st.button("Ask Question")

if ask_question:
    if uploaded_file:
        # 1. Upload the PDF
        upload_pdf(uploaded_file)

        # 2. Load the PDF
        file_path = "pdfs/" + uploaded_file.name
        documents = load_pdf(file_path)

        # 3. Create chunks
        text_chunks = create_chunks(documents)

        # 4. Create vector database
        faiss_db = create_vector_db(text_chunks, "deepseek-r1:1.5b")

        # 5. Retrieve documents and answer query
        st.chat_message("user").write(user_query)
        retrieved_docs = retrieve_docs(user_query, faiss_db)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        st.chat_message("Assistant").write(response.content)
    else:
        st.error("Please upload a PDF file before asking a question.")