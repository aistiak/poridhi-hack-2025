import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("intent_classifier")

# === Load label encoder ===
# Make sure this matches the label order during training
df_intents = pd.read_csv("intent_data.csv")  # This file must contain the original "intent" column
label_encoder = LabelEncoder()
label_encoder.fit(df_intents["intent"])

# === Load ONNX model ===
onnx_model_path = "intent_classifier.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# === Prepare query for inference ===
query = "find a wireless mouse"

# Tokenize input (return PyTorch tensors and convert to numpy)
inputs = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
input_ids = inputs["input_ids"].numpy()
attention_mask = inputs["attention_mask"].numpy()

# === Run inference with ONNX ===
ort_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask
}
ort_outputs = ort_session.run(None, ort_inputs)
logits = ort_outputs[0]

# === Get predicted intent ===
predicted_label = np.argmax(logits, axis=1)[0]
predicted_intent = label_encoder.inverse_transform([predicted_label])[0]

# === Output ===
print(f"Query: {query}")
print(f"Predicted Intent: {predicted_intent}")
