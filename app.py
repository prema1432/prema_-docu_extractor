import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGroq(temperature=0, groq_api_key="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",
               model_name="mixtral-8x7b-32768")

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)


# Function to create ChromaDB from PDF
def create_chroma_db(pdf_file, keywords):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)
    print("all_splits", all_splits)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    # Define prompt for question-answering
    # prompt = hub.pull("rlm/rag-prompt")

    #     template = """You are an intelligent assistant designed to extract structured data from text documents. Analyze the provided content and extract details corresponding to the given keywords (e.g., "NAME," "EMAIL," "PHONE NUMBER"). Provide the extracted data in a clear format, and if any keyword data is missing, state "Not found.
    #
    #
    #     {context}
    #
    #     Question: The keywords I want to extract are:
    #  {question}
    # Please extract the data for these keywords.
    #     Helpful Answer:"""
    template = """You are an intelligent assistant designed to analyze text documents and provide actionable insights based on the specified keywords. For each keyword, extract relevant information from the provided context, analyze it, and offer concise solutions or recommendations without repeating phrases like "The text mentions."

    Context:
    {context}

    Keywords to analyze:
    {question}

    For each keyword, please provide:
    - Keyword: [concise solution or recommendation]


    Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
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

    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # db = Chroma.from_documents(documents, embeddings)

    # return db
    # return documents
    #     send_question = f"""
    # The keywords I want to extract are:
    # {keywords}
    # Please extract the data for these keywords."""
    response = graph.invoke({"question": keywords})
    print(response["answer"])
    return response["answer"]


# Function to search in ChromaDB
# def search_chroma_db(db, keywords):
#     results = db.similarity_search(keywords)
#     return results


# Streamlit UI
st.title("PDF Keyword Search with ChromaDB")

# Input for OpenAI API Key
# openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Input for keywords
keywords = st.text_input("Enter keywords to search:")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Process"):
    if keywords and pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        print("temp_file_pathtemp_file_path", temp_file_path)
        # Create ChromaDB
        db = create_chroma_db(temp_file_path, keywords)
        # st.success("ChromaDB created successfully!")

        # Search in the ChromaDB
        results = db
        # results = search_chroma_db(db, keywords)

        if results:
            st.write("Search Results:")
            st.write(results)
            # st.write("Search Results:")
        #     for result in results:
        #         st.write(result.page_content)
        else:
            st.write("No results found.")

        # Clean up: Delete ChromaDB
        # del db
        # os.remove(temp_file_path)
        # st.success("ChromaDB deleted successfully!")
    else:
        st.error("Please provide all inputs.")
