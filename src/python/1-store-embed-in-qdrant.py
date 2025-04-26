import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 1. Load and prepare product data
print('> reading the data from csv')
# df = pd.read_csv("products.csv")
# df["text"] = df["name"] + " " + df["description"] + " " + df["category"]
# print(df)
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["hack"]  # Use your actual database name
collection = db["products"]  # Use your actual collection name

# Fetch all products from MongoDB
mongo_products = list(collection.find({}).skip(0).limit(1000))

# Convert MongoDB data to DataFrame
if mongo_products:
    # Create DataFrame from MongoDB data
    df = pd.DataFrame(mongo_products)
    # Drop MongoDB's _id field if it exists
    # if "_id" in df.columns:
    #     df = df.drop("_id", axis=1)
    # Create text field for embedding
    df["title"] = str(df["title_left"])
    df["description"] = str(df["description_left"] +   "  " + df["description_right"])
    df["category"] = str(df["category_left"])
    
    df["text"] = df["title"] + " " + df["description"] + " " + df["category"]
    print(f"Loaded {len(df)} products from MongoDB")
else:
    print("No products found in MongoDB, using CSV data instead")
    exit(0)
    # Keep the existing CSV loading code as fallback
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
# qdrant = QdrantClient("http://47.128.215.131:6333")

print('> connected to qdrant')

embedding_dim = embeddings.shape[1]  # e.g., 384
# Define collection name
collection_name = "product_embeddings"

# Check if collection exists, create only if it doesn't
collections = qdrant.get_collections().collections
collection_exists = any(collection.name == collection_name for collection in collections)

if not collection_exists:
    print(f"> creating collection '{collection_name}'")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
else:
    print(f"> collection '{collection_name}' already exists, using it")

# 4. Store embeddings in Qdrant
print('> storing embeddings')

points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "id": str(row["_id"]),
            "title": row["title"],
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
    print(f"Product: {hit.payload['title']}, Category: {hit.payload['category']}, Score: {hit.score}")
    print(hit.payload)