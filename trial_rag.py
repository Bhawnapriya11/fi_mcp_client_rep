# import numpy as np
# from google.cloud import language_v1
# from google.oauth2 import service_account
# from langchain.text_splitter import CharacterTextSplitter
# from sklearn.metrics.pairwise import cosine_similarity
# import fitz  # PyMuPDF

# # Google Cloud Authentication and API setup
# PROJECT_ID = "agenticaigcp"
# KEY_PATH = "/Users/bhawna/Downloads/agenticaigcp-b6166246b712.json"  # Service account key for Google Cloud authentication
# LOCATION = "us-central1"

# # Initialize Google Cloud's Natural Language API
# credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
# client = language_v1.LanguageServiceClient(credentials=credentials)

# # Example function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     pdf_document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text("text")
#     return text

# # Function to get text embeddings using Google Cloud's Natural Language API
# def get_embeddings(texts):
#     # Create a document for the API to process
#     document = language_v1.Document(content=texts, type_=language_v1.Document.Type.PLAIN_TEXT)
#     # Get embeddings (you can use analyze_sentiment, entities, or other methods for NLP tasks)
#     response = client.analyze_sentiment(document=document)
#     return response.document_sentiment.score  # Or use other types like entities or syntax for embeddings

# # Chunking document into smaller parts
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# # Extract text from the document
# pdf_path = "/Users/bhawna/Downloads/2024INB00507948MBR_Form16_PART_B.pdf"
# document_text = extract_text_from_pdf(pdf_path)
# chunks = text_splitter.split_text(document_text)

# # Generate embeddings for the document chunks
# document_embeddings = [get_embeddings(chunk) for chunk in chunks]

# # Function to perform semantic search and retrieve the most relevant document
# def retrieve_relevant_document(query, embeddings, chunks):
#     query_embedding = get_embeddings(query)  # Embed the query text
#     cosine_similarities = cosine_similarity([query_embedding], embeddings)  # Compare cosine similarities
#     best_match_idx = np.argmax(cosine_similarities)  # Get the most relevant document
#     return chunks[best_match_idx]

# # Example query
# query = "What is total salary in Mercedes benz?"

# # Retrieve the most relevant document based on cosine similarity
# relevant_document = retrieve_relevant_document(query, document_embeddings, chunks)
# print(f"Most Relevant Chunk: {relevant_document}")

# # You can use **Google's pre-trained model** for generation (T5, BERT) or use Vertex AI if you have access to a model.
# # For example, using Vertex AI or other NLP models for the generation step:
# # (Replace with a call to Vertex AI if needed)

# # Generate a response from the relevant document (answering the query)
# def generate_answer_with_nlp(query, context):
#     # Here we use a placeholder model for generation. You can replace this with a model like T5.
#     # Vertex AI or HuggingFace could be used to generate the answer based on the context.
#     return f"Answer generated based on the context: {context}"

# answer = generate_answer_with_nlp(query, relevant_document)
# print(f"Generated Answer: {answer}")
import os
from dotenv import load_dotenv
# Import PyPDFLoader for PDF support
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure your GOOGLE_API_KEY is set in your environment variables or a .env file.
# You can get one from Google Cloud Console or AI Studio.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment variables.")
    exit()

# Directory to store your Minecraft server knowledge base documents (PDFs)
KNOWLEDGE_BASE_DIR = "minecraft_knowledge_base"
# Directory to persist the Chroma vector store
CHROMA_DB_DIR = "chroma_db"

# --- 1. Prepare Knowledge Base Directory ---
def prepare_knowledge_base_directory(directory: str):
    """
    Ensures the knowledge base directory exists and instructs the user
    to place their PDF files here.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Please place your PDF knowledge base files into the '{directory}' directory.")
    print(f"Current files in '{directory}': {os.listdir(directory)}")
    if not os.listdir(directory):
        print("Warning: No PDF files found in the knowledge base directory. The RAG agent will have no context.")
        print("Consider adding some dummy PDFs or your actual server documentation PDFs.")

# --- 2. Load Documents (now supporting PDFs) ---
def load_documents(directory: str):
    """Loads all PDF documents from the specified directory."""
    print(f"Loading PDF documents from '{directory}'...")
    # Use DirectoryLoader with PyPDFLoader for PDF files
    # glob="**/*.pdf" ensures only PDF files are loaded
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF documents.")
    if not documents:
        print("No documents were loaded. Please ensure your PDF files are in the specified directory.")
    return documents

# --- 3. Split Documents ---
def split_documents(documents):
    """Splits documents into smaller chunks for better retrieval."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# --- 4. Create and Persist Vector Store (ChromaDB) ---
def create_vector_store(chunks, db_directory: str):
    """
    Creates or loads a Chroma vector store from document chunks.
    It persists the store to disk, so it doesn't need to be re-indexed every time.
    """
    print(f"Initializing embeddings model...")
    # Using GoogleGenerativeAIEmbeddings for Vertex AI compatibility
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the Chroma DB already exists
    if os.path.exists(db_directory) and os.listdir(db_directory):
        print(f"Loading existing Chroma DB from '{db_directory}'...")
        vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    else:
        print(f"Creating new Chroma DB and persisting to '{db_directory}'...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_directory
        )
        vector_store.persist()
        print("Chroma DB created and persisted.")
    return vector_store

# --- 5. Set up the RAG Chain ---
def setup_rag_chain(vector_store):
    """Sets up the RAG (Retrieval Augmented Generation) chain."""
    print("Setting up RAG chain...")

    # Initialize the LLM (using Google's Gemini-Pro model)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Define the prompt template for the LLM
    # The 'context' variable will be populated by the retriever
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant for a Minecraft server.
    Answer the user's question based only on the provided context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {input}
    """)

    # Create a document chain to combine retrieved documents with the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain setup complete.")
    return retrieval_chain

# --- Main Execution ---
def main():
    # 1. Prepare knowledge base directory and instruct user
    prepare_knowledge_base_directory(KNOWLEDGE_BASE_DIR)

    # 2. Load documents (PDFs)
    documents = load_documents(KNOWLEDGE_BASE_DIR)

    if not documents:
        print("No documents loaded. Please add PDF files to the knowledge base directory.")
        return # Exit if no documents are found

    # 3. Split documents
    chunks = split_documents(documents)

    # 4. Create or load vector store
    vector_store = create_vector_store(chunks, CHROMA_DB_DIR)

    # 5. Set up RAG chain
    rag_chain = setup_rag_chain(vector_store)

    print("\n--- Minecraft Server RAG Agent Ready ---")
    print("Type your questions about the server. Type 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            print("Exiting RAG agent. Goodbye!")
            break

        print("Thinking...")
        try:
            # Invoke the RAG chain with the user's question
            response = rag_chain.invoke({"input": user_input})
            print("\nAgent Response:")
            print(response["answer"])
            # Optionally, you can print the source documents that were used
            # print("\n--- Source Documents ---")
            # for doc in response["context"]:
            #     print(f"Source: {doc.metadata.get('source', 'N/A')}")
            #     print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
            #     print("-" * 20)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your GOOGLE_API_KEY is valid, you have internet access, and valid PDF files.")

if __name__ == "__main__":
    main()