import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct



fine_tuned_model = SentenceTransformer("fine_tuned_model")

print(fine_tuned_model)



# search in qdrant


qdrant = QdrantClient("http://localhost:6333")

print('> connected to qdrant')


query_text = "mouse"
print('> query qdrant {}',query_text)
query_embedding = fine_tuned_model.encode([query_text])[0]

print('> searching results')
collection_name = "product_embeddings"

search_result = qdrant.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=3  # Top 3 similar products
)

# Print results
print("\nSearch Results:")
for hit in search_result:
    print(f"Product: {hit.payload['name']}, Category: {hit.payload['category']}, Score: {hit.score}")
    print(hit.payload)