import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
import os

from langchain_community.document_loaders import PyPDFLoader


# Function to create ChromaDB from PDF
def create_chroma_db(pdf_file, openai_api_key):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # db = Chroma.from_documents(documents, embeddings)

    # return db
    return documents


# Function to search in ChromaDB
# def search_chroma_db(db, keywords):
#     results = db.similarity_search(keywords)
#     return results


# Streamlit UI
st.title("PDF Keyword Search with ChromaDB")

# Input for OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Input for keywords
keywords = st.text_input("Enter keywords to search:")

# File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Process"):
    if openai_api_key and keywords and pdf_file:
        # Create ChromaDB
        db = create_chroma_db(pdf_file, openai_api_key)
        st.success("ChromaDB created successfully!")

        # Search in the ChromaDB
        results = db
        # results = search_chroma_db(db, keywords)

        if results:
            st.write("Search Results:")
        #     for result in results:
        #         st.write(result.page_content)
        else:
            st.write("No results found.")

        # Clean up: Delete ChromaDB
        # del db
        st.success("ChromaDB deleted successfully!")
    else:
        st.error("Please provide all inputs.")