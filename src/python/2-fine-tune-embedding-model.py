import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
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

# # Export to ONNX using optimum
# from optimum.exporters.onnx import main_export
# main_export(
#     model_name_or_path="fine_tuned_model",
#     output="fine_tuned_model_onnx",
#     task="feature-extraction",
#     model=transformers_model,
#     tokenizer=tokenizer
# )
# print("Fine-tuned model exported to ONNX at 'fine_tuned_model_onnx'")

# # 7. Load ONNX model for inference
# ort_model = ORTModelForFeatureExtraction.from_pretrained("fine_tuned_model_onnx", provider="CPUExecutionProvider")
# tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model_onnx")
# print(tokenizer)