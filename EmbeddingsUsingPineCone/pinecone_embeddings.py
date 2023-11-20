import json
import os
import pinecone
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to config.json
filepath = os.path.join(current_dir, 'config.json')

# load the api keys
def load_config():
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

config = load_config()

# Now you can access your keys like this:
openai_api_key = config['openai_api_key']
pinecone_api_key = config['pinecone_api_key'] 

#Configure the openai's key
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Fetch from the pinecone
# initialize pinecone and fetch the data
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