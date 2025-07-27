from pinecone import Pinecone, ServerlessSpec


# Initialize Pinecone (you need to get your API key from the Pinecone dashboard)
api_key=""

pc= Pinecone(api_key=api_key)
# Create a new index (or use an existing one)
index_name = "example-index"

existing_indexes = pc.list_indexes()

if index_name not in existing_indexes.names():
    # Create an index with the desired specifications
    pc.create_index(
        name=index_name,
        dimension=512,  # Set dimension based on your embeddings
        metric='euclidean',  # You can choose other metrics like 'cosine'
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-east1-gcp'  # Choose the region that best fits your use case
        )
    )

# Connect to the index
index = pc.index(index_name)

# Upsert example vector data (embeddings)
vector = [0.1, 0.2, 0.3, 0.4]  # Replace with your actual vector
metadata = {"source": "document1"}
index.upsert([(1, vector, metadata)])

# Query the index for the top-k most similar vectors
query_vector = [0.2, 0.3, 0.4, 0.5]  # Your query vector
result = index.query(query_vector, top_k=5)

print(result)
