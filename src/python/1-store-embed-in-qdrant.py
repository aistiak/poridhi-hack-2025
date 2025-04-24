import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 1. Load and prepare product data
print('> reading the data from csv')
df = pd.read_csv("products.csv")
df["text"] = df["name"] + " " + df["description"] + " " + df["category"]
print(df)
# 2. Generate embeddings using Sentence-Transformers
print('> loading model')
model = SentenceTransformer('all-MiniLM-L6-v2')  # CPU-friendly model
print('> model loaded')
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
print('> embeddings generated')
print(embeddings)
# 3. Connect to Qdrant and create a collection
print('> connecting to qdrant')

qdrant = QdrantClient("http://localhost:6333")

print('> connected to qdrant')

embedding_dim = embeddings.shape[1]  # e.g., 384

# Recreate collection (drops existing one, use carefully)
collection_name = "product_embeddings"
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

# 4. Store embeddings in Qdrant
print('> storing embeddings')

points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "id": int(row["id"]),
            "name": row["name"],
            "description": row["description"],
            "category": row["category"]
        }
    )
    for idx, (embedding, row) in enumerate(zip(embeddings, df.to_dict("records")))
]

qdrant.upsert(collection_name=collection_name, points=points)
print('> embeddings stored ')
print(f"Stored {len(points)} products in Qdrant.")

# 5. Retrieve similar products for a query

query_text = "mouse"
print('> query qdrant {}',query_text)
query_embedding = model.encode([query_text])[0]

print('> searching results')
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