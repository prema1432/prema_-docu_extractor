__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

load_dotenv()
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# vector_store = InMemoryVectorStore(embeddings

default_prompt = """You are an intelligent assistant designed to analyze text documents and provide actionable insights based on the specified keywords. For each keyword, extract relevant information from the provided context, analyze it, and offer concise solutions or recommendations without repeating phrases like "The text mentions."

    Context:
    {context}

    Keywords to analyze:
    {question}

    For each keyword, please provide:
    - concise solution or recommendation in the our database
    Helpful Answer:"""

def create_chroma_db(pdf_file, keywords, random_uuid,user_prompt):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)
    st.progress(70)

    # # Index chunks
    # _ = Chroma.add_documents(documents=all_splits)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings, collection_name=random_uuid,
                                        persist_directory=f"./chromadb/{random_uuid}")
    # template = """You are an intelligent assistant designed to analyze text documents and provide actionable insights based on the specified keywords. For each keyword, extract relevant information from the provided context, analyze it, and offer concise solutions or recommendations without repeating phrases like "The text mentions."
    #
    # Context:
    # {context}
    #
    # Keywords to analyze:
    # {question}
    #
    # For each keyword, please provide:
    # - concise solution or recommendation in the our database only.
    # Helpful Answer:"""
    template = user_prompt
    prompt = PromptTemplate.from_template(template)

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": keywords})
    print(response["answer"])
    return response["answer"]


st.title("PDF Keyword Search")
keywords = st.text_input("Enter keywords to search:")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
user_prompt = st.text_area("Enter your custom prompt:", value=default_prompt, height=350)

if st.button("Process"):
    if keywords and pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        print("temp_file_pathtemp_file_path", temp_file_path)
        random_uuid = str(uuid.uuid4())
        with st.spinner("Uploading and extracting the file..."):
            st.progress(30)
            results = create_chroma_db(temp_file_path, keywords, random_uuid,user_prompt)

        st.success("Processing complete!")
        st.progress(100)

        if results:
            st.write("Search Results:")
            st.write(results)
            st.empty()
        else:
            st.write("No results found.")

        directory_path = f"./chromadb/{random_uuid}"

        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)  
            print(f"Deleted directory: {directory_path}")
        else:
            print(f"Directory does not exist: {directory_path}")
    else:
        st.error("Please provide all inputs.")
