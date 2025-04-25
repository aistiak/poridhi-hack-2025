import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reload model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("intent_classifier")
tokenizer = AutoTokenizer.from_pretrained("intent_classifier")


from optimum.exporters.onnx import main_export
main_export(
    model_name_or_path="intent_classifier",
    output="fine_tuned_model_onnx2",
    task="text-classification",
    # model=model,
    tokenizer=tokenizer
)
print("Fine-tuned model exported to ONNX at 'fine_tuned_model_onnx2'")

# 7. Load ONNX model for inference
from optimum.onnxruntime import ORTModelForFeatureExtraction
ort_model = ORTModelForFeatureExtraction.from_pretrained("fine_tuned_model_onnx2", provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model_onnx2")
print(tokenizer)