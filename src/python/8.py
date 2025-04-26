import torch
import onnxruntime as ort
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

ort_model = ORTModelForFeatureExtraction.from_pretrained("out_models/embedding_model_onnx", provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained("out_models/embedding_model_onnx")


# 8. Generate embeddings for products
def generate_embeddings(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Generate embeddings using the ONNX model
    with torch.no_grad():
        outputs = ort_model(**inputs)
    
    # Get the embeddings from the model output (mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

text = "camera"
product_embeddings = generate_embeddings([text])
print(f"Generated {len(product_embeddings)} embeddings with dimension {product_embeddings.shape[1]}")