# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain.embeddings import VertexAIEmbeddings
# from langchain.llms import VertexAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.schema import Document
# from langchain.chains import RetrievalQA
# from google.cloud import storage
# import vertexai

# import io
# import os

# # CONFIG
# BUCKET_NAME = "test-bucket-bhawna-unique-123456"
# PREFIX = "trial_docs/"  # GCS folder
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

# # Init FastAPI
# app = FastAPI()

# # Init Vertex AI
# vertexai.init(project="agenticaigcp", location="us-central1")
# llm = VertexAI(model_name="text-bison", temperature=0.2)
# embedding = VertexAIEmbeddings()

# # Pydantic model
# class Query(BaseModel):
#     query: str

# # Utility to load all docs from GCS
# def load_documents_from_gcs(bucket_name, prefix):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=prefix)

#     texts = []
#     for blob in blobs:
#         content = blob.download_as_text()
#         texts.append(content)
#     return texts

# # Split and embed
# def get_retriever_from_gcs():
#     raw_docs = load_documents_from_gcs(BUCKET_NAME, PREFIX)
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
#     )
#     chunks = splitter.create_documents(raw_docs)

#     vectorstore = DocArrayInMemorySearch.from_documents(chunks, embedding)
#     return vectorstore.as_retriever()

# # @app.post("/rag")
# # def get_rag_response(body: Query):
# #     retriever = get_retriever_from_gcs()
# #     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# #     result = qa.run(body.query)
# #     return {"result": result}

# retriever = get_retriever_from_gcs()
# qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
# query="what is the policy?"
# response=qa.run(query)
# print("rag response",response)

# c

# import vertexai
# from langchain_google_vertexai import ChatVertexAI
# import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/bhawna/Desktop/agentic_ai_day/agenticai/agenticaiGCP-b6166246b712.json"

# # vertexai.init(project="agenticaigcp", location="us-central1")

# # llm = ChatVertexAI(model_name="chat-bison@002", temperature=0.3)
# # response = llm.invoke("Tell me about Google Cloud.")
# # print(response.content)
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI

# os.environ["GOOGLE_API_KEY"] = "-UmTjw"

# model=ChatGoogleGenerativeAI(model='gemini-1.5.pro')
# result=model.invoke('what is chatgpt?')
# print(result.content)

'''
working code for the gemini answering

import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.oauth2 import service_account

# Config
PROJECT_ID = "agenticaigcp"
LOCATION = "us-central1"
KEY_PATH = "/Users/bhawna/Desktop/agentic_ai_day/agenticai/agenticaigcp-b6166246b712.json"
MODEL_NAME = "gemini-2.5-flash"

# Authenticate and initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Initialize Gemini model and ask questions
model = GenerativeModel(MODEL_NAME)
questions = ["What is Pradhan mantri mudra yojna?"]
for q in questions:
    response = model.generate_content(q).candidates[0].content.parts[0].text
    print(f"Q: {q}\nA: {response}\n")
'''
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")  # Extract text from each page
    return text

# Example usage
pdf_path = "/Users/bhawna/Downloads/2024INB00507948MBR_Form16_PART_B.pdf"
document_text = extract_text_from_pdf(pdf_path)
#print(document_text)  # Print first 500 characters for a preview
print('*********************')

# Initialize the splitter and chunk the document text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(document_text)

print(len(chunks))
# Print the first chunk
print(chunks[0],'chunks data')

from google.cloud import aiplatform
from google.oauth2 import service_account

# Initialize Vertex AI
PROJECT_ID = "agenticaigcp"
LOCATION = "us-central1"
KEY_PATH = "/Users/bhawna/Downloads/agenticaigcp-b6166246b712.json"
MODEL_NAME = "gemini-2.5-flash"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Initialize Gemini model
model = GenerativeModel(MODEL_NAME)

# Function to generate embeddings using Vertex AI's Gemini model
def generate_embeddings_with_vertex(texts):
    model = aiplatform.Model("projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_MODEL_ENDPOINT")
    embeddings = model.predict(instances=texts)
    return embeddings

# Example: Generate embeddings for document chunks
embeddings = generate_embeddings_with_vertex(chunks)


