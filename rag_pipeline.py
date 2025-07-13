from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b") 

# Retrieve Documents from the vector database
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context ="\n\n".join([doc.page_content for doc in documents])
    return context

# Answer the question using the LLM
custome_prompt = """
Imagine I'm a helpful teacher and the text below is our study material for today.
My job is to answer your question using **only** the information from this material. I will explain the answer in a very simple and clear way.
If our study material doesn't have the answer, I will honestly tell you, "That information isn't in the text we have." I promise not to guess or make things up.
Okay, let's begin!

Your Question: {question}
Our Study Material (Context): {context}
My Helpful Answer:
"""

def answer_query(documents, model, query): 
    context = get_context(documents)
    prompt =  ChatPromptTemplate.from_template(custome_prompt)
    chain = prompt | model
    return chain.invoke({"question": query,"context": context}) 
