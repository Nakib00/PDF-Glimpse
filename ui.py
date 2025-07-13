from rag_pipeline import answer_query, retrieve_docs, llm_model
import streamlit as st
from vector_database import upload_pdf, load_pdf, create_chunks, create_vector_db

# Set the page title and icon
st.set_page_config(
    page_title="PDF Glimpse",
    page_icon="📄"
)

st.title("PDF Glimpse")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
user_query = st.text_area("Enter your query here:", height=150, placeholder="Type your question about the PDF...")
ask_question = st.button("Ask Question")

if ask_question:
    if uploaded_file:
        # Process the PDF and get documents
        upload_pdf(uploaded_file)
        file_path = "pdfs/" + uploaded_file.name
        documents = load_pdf(file_path)
        text_chunks = create_chunks(documents)
        faiss_db = create_vector_db(text_chunks, "deepseek-r1:1.5b")
        retrieved_docs = retrieve_docs(user_query, faiss_db)

        # Get the response (with thinking) from the pipeline
        response_with_thinking = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        # Separate the thinking from the final answer
        thinking_text = None
        final_answer = response_with_thinking

        if "<think>" in response_with_thinking and "</think>" in response_with_thinking:
            start_index = response_with_thinking.find("<think>") + len("<think>")
            end_index = response_with_thinking.find("</think>")
            thinking_text = response_with_thinking[start_index:end_index].strip()
            final_answer = response_with_thinking[end_index + len("</think>"):].strip()

        # Display the user's message
        st.chat_message("user").write(user_query)
        
        # Display the final answer
        with st.chat_message("Assistant"):
            # If there's a "thinking" part, show it in an expander
            if thinking_text:
                with st.expander("🤔 Thinking..."):
                    st.info(thinking_text)
            
            st.write(final_answer)

    else:
        st.error("Please upload a PDF file before asking a question.")