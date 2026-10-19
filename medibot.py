# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_groq import ChatGroq


# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
#         #HF_TOKEN=os.environ.get("HF_TOKEN")  

#         #TODO: Create a Groq API key and add it to .env file
        
#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=ChatGroq(
#                     model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
#                     temperature=0.0,
#                     groq_api_key=os.environ["GROQ_API_KEY"],
#                 ),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()

import os
from unittest import result
from urllib import response
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "input"]
    )
    return prompt

def main():

    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:

        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.

If you dont know the answer, just say that you dont know, dont try to make up an answer.

Dont provide anything out of the given context.

Context: {context}

Question: {input}

Start the answer directly. No small talk please.
"""

        try:

            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # -----------------------------
            # Groq LLM
            # -----------------------------

            # llm = ChatGroq(
            #     model_name="openai/gpt-oss-120b",
            #     temperature=0.0,
            #     groq_api_key=os.environ["GROQ_API_KEY"],
            # )
            api_key = os.environ.get("GROQ_API_KEY", "")

            st.write("API key loaded:", bool(api_key))
            st.write("API key starts with:", api_key[:4] if api_key else "MISSING")
            st.write("API key length:", len(api_key))

            llm = ChatGroq(
                model_name="openai/gpt-oss-120b",
                temperature=0.0,
                groq_api_key=api_key,
            )

            # -----------------------------
            # Prompt
            # -----------------------------

            prompt_template = set_custom_prompt(
                CUSTOM_PROMPT_TEMPLATE
            )

            # -----------------------------
            # Retriever
            # -----------------------------

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            # -----------------------------
            # Document chain
            # -----------------------------

            question_answer_chain = create_stuff_documents_chain(
                llm,
                prompt_template
            )

            # -----------------------------
            # Retrieval chain
            # -----------------------------

            qa_chain = create_retrieval_chain(
                retriever,
                question_answer_chain
            )

            # -----------------------------
            # Ask question
            # -----------------------------

            response = qa_chain.invoke(
                {
                    "input": prompt
                }
            )

            result = response["answer"]

            source_documents = response["context"]

# -----------------------------
# Format sources
# -----------------------------

            sources_text = "\n\n**Sources:**\n"

            for i, doc in enumerate(source_documents, start=1):
                page = doc.metadata.get("page_label", doc.metadata.get("page", "Unknown"))
                source = doc.metadata.get("source", "Unknown source")

            source_name = os.path.basename(source)

            sources_text += f"- 📄 {source_name} — Page {page}\n"

            result_to_show = result + sources_text

            st.chat_message("assistant").markdown(result_to_show)

            st.session_state.messages.append(
                {
                "role": "assistant",
                "content": result_to_show
        }
     )
        except Exception as e:

            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()