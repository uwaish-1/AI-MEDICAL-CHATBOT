import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

#  1 - setup LLM (MIstral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    return llm

#  2 - connect LLM with FAISS vectorstore

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db= FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


#  3- create chain

# qa_chain=RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k':3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

llm = load_llm(HUGGINGFACE_REPO_ID)

prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

qa_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# Now invoke with a single query
user_query=input("Write Query Here: ")
response = qa_chain.invoke({
    "input": user_query
})

print("RESULT: ", response["answer"])
print("SOURCE DOCUMENTS: ", response["context"])
