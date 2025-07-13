from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st



uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

user_query = st.text_area("Enter your query here:", height=150, placeholder="Type your question about the PDF...")

ask_question = st.button("Ask Question")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)
        
        retrieve_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieve_docs, model=llm_model, query=user_query)
        st.chat_message("Assistant").write(response)
    else:
        st.error("Please upload a PDF file before asking a question.")