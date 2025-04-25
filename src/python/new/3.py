# GET INTENT CLASS PREDICTION FROM MODEL 

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification  # Changed from FeatureExtraction
import torch

# Load the ONNX model and tokenizer
# model_dir = "fine_tuned_model_onnx2"
model_dir = "intent_classifier_onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Sample input text
text = "detail of shoes"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Convert to numpy arrays for ONNX Runtime
onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

# Get predictions
outputs = ort_model(**onnx_inputs)
logits = outputs.logits

# Get predicted class (intent)
predicted_class = torch.argmax(torch.from_numpy(logits), dim=1).item()

print(f"Predicted intent class: {predicted_class}")