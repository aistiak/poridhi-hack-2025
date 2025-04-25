# finetune embedding model and save in onnx

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
from torch.utils.data import DataLoader
from random import choice
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort
from pathlib import Path

# 1. Load and prepare product data
df = pd.read_csv("products.csv")
df["text"] = df["name"] + " " + df["description"] + " " + df["category"]

# Connect to MongoDB and get products data
# from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")
# db = client["products_db"]  # Use your actual database name
# collection = db["products"]  # Use your actual collection name

# # Fetch all products from MongoDB
# mongo_products = list(collection.find({}))

# # Convert MongoDB data to DataFrame
# if mongo_products:
#     # Create DataFrame from MongoDB data
#     df = pd.DataFrame(mongo_products)
#     # Drop MongoDB's _id field if it exists
#     if "_id" in df.columns:
#         df = df.drop("_id", axis=1)
#     # Create text field for embedding
#     df["text"] = df["name"] + " " + df["description"] + " " + df["category"]
#     print(f"Loaded {len(df)} products from MongoDB")
# else:
#     print("No products found in MongoDB, using CSV data instead")
#     # Keep the existing CSV loading code as fallback

# 2. Create triplet training examples for fine-tuning
train_examples = []
for idx, row in df.iterrows():
    anchor = row["text"]
    positive = row["text"]  # Simplified; ideally use similar product
    negative = choice(df[df["category"] != row["category"]]["text"].tolist())
    train_examples.append(InputExample(texts=[anchor, positive, negative]))

# 3. Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # CPU-friendly

# 4. Prepare DataLoader and triplet loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model=model)

# 5. Fine-tune and save the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="fine_tuned_model",
    show_progress_bar=True
)
print("Fine-tuned model saved to 'fine_tuned_model'")

# 6. Export fine-tuned model to ONNX
# Load the fine-tuned model as a Transformers model
transformers_model = AutoModel.from_pretrained("fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")

# Export to ONNX using optimum
from optimum.exporters.onnx import main_export
main_export(
    model_name_or_path="fine_tuned_model",
    output="fine_tuned_model_onnx",
    task="feature-extraction",
    # model=transformers_model,
    tokenizer=tokenizer
)
print("Fine-tuned model exported to ONNX at 'fine_tuned_model_onnx'")
# 7. Load ONNX model for inference and generate embeddings
ort_model = ORTModelForFeatureExtraction.from_pretrained("fine_tuned_model_onnx", provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model_onnx")

# 8. Generate embeddings for products
# def generate_embeddings(texts):
#     # Tokenize the input texts
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
#     # Generate embeddings using the ONNX model
#     with torch.no_grad():
#         outputs = ort_model(**inputs)
    
#     # Get the embeddings from the model output (mean pooling)
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.numpy()

# # Generate embeddings for all products
# product_embeddings = generate_embeddings(df["text"].tolist())
# print(f"Generated {len(product_embeddings)} embeddings with dimension {product_embeddings.shape[1]}")

# # 9. Store embeddings in Qdrant
# qdrant = QdrantClient("http://localhost:6333")

# # Create or recreate collection
# collection_name = "fine_tuned_product_embeddings"
# embedding_dim = product_embeddings.shape[1]

# qdrant.recreate_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
# )

# # Prepare points for Qdrant
# points = [
#     PointStruct(
#         id=idx,
#         vector=embedding.tolist(),
#         payload={
#             "id": int(row["id"]),
#             "name": row["name"],
#             "description": row["description"],
#             "category": row["category"]
#         }
#     )
#     for idx, (embedding, row) in enumerate(zip(product_embeddings, df.to_dict("records")))
# ]

# # Upload to Qdrant
# qdrant.upsert(collection_name=collection_name, points=points)
# print(f"Stored {len(points)} product embeddings in Qdrant collection '{collection_name}'")