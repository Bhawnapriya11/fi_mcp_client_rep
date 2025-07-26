from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Use a smaller, faster model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example documents
documents = ["This is a test document.", "Another test document.", "This is an example of text."]
embeddings = model.encode(documents, show_progress_bar=True)

# Example query
query = "test document"
query_embedding = model.encode([query])

# Compute cosine similarity between query and documents
cosine_similarities = cosine_similarity(query_embedding, embeddings)

# Get the index of the top matching document
top_match_index = np.argmax(cosine_similarities)
print("Top matching document:", documents[top_match_index])
