import json
import os
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

import streamlit as st

# Print current working directory
st.title(f"Current Working Directory: {os.getcwd()}")

# Assuming this script is located in the EmbeddingsUsingPineCone directory
current_dir = os.path.dirname(__file__)

# Get the absolute path of the project root by navigating up one directory
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))

# Join the absolute path with .env
dotenv_path = os.path.join(project_root, '.env')

# Print whether .env file exists
st.title(f".env File Exists: {os.path.exists(dotenv_path)}")

#Load the API keys from the .env file
load_dotenv(dotenv_path)


#Configure the openai's key
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

st.title(openai_api_key)
st.title(pinecone_api_key)

# Configure the OpenAI's Ada model for embeddings
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Fetch from the pinecone
# initialize pinecone and fetch the data
os.environ["OPENAI_API_KEY"] = pinecone_api_key
pinecone.init(
    api_key=pinecone_api_key,  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)
index_name = "cp1-test2"

#load the vector db
docsearch = Pinecone.from_existing_index(index_name, embeddings)

def get_similar_docs(query, k=1, score=False):
    while True:
        try:
            if score:
                similar_docs = docsearch.similarity_search_with_score(query, k=k)
            else:
                similar_docs = docsearch.similarity_search(query, k=k)

            # Extract page_content from the first Document in the list
            content = similar_docs[0].page_content

            # Find the first occurrence of " and the next ",
            start_index = content.find('"') + 1
            end_index = content.find('",', start_index)

            # Extract the content between the first " and the next ",
            extracted_content = content[start_index:end_index]

            print(f"\nExtracted content: {extracted_content}")
            return extracted_content

        except Exception as e:
            # Handle Pinecone API exceptions
            print(f"Pinecone API exception: {e}")

            # Optionally, wait and retry
            wait_time = 5  # Adjust the wait time as needed
            print(f"Waiting for {wait_time} seconds.")
            time.sleep(wait_time)

            return "I don't understand what you are asking. Please rephrase your prompt."

# # Example usage
# query = "How can students involve in a major cheating in examinations?"
# similar_docs = get_similar_docs(query)
# print(similar_docs)