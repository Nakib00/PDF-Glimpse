from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Retrieve Documents from the vector database
def retrieve_docs(query, db):
    return db.similarity_search(query)

def get_context(documents):
    context ="\n\n".join([doc.page_content for doc in documents])
    return context

# NEW: A more detailed prompt that encourages "thinking"
custom_prompt_with_thinking = """
First, think step-by-step and explain your reasoning within `<think>` tags. Analyze the user's question and the provided "Study Material" to formulate your answer. Your thought process should be clear and logical.

After your thinking process, provide a final, concise answer to the user. Base your answer **only** on the information from the "Study Material". If the material doesn't contain the answer, state that clearly.

Your Question: {question}
Our Study Material (Context): {context}

My Helpful Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt =  ChatPromptTemplate.from_template(custom_prompt_with_thinking)
    chain = prompt | model
    # The response will now contain both the <think> block and the final answer
    response_content = chain.invoke({"question": query,"context": context}).content
    return response_content