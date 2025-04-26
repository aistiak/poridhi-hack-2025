# FINE TUNE CLASSIFIER MODEL AND SAVE TO ONNX

import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Load product data
df_products = pd.read_csv("products.csv")

# 2. Define query templates for each intent
## search , filter category , filter price rance , is stock , out of scope 
templates = {
    "search_products": [
        "Find a {name}",
        "Search for {description_keywords}",
        "Show me {name} for {category}",
        "Look for {description_keywords}"
    ],
    "filter_category": [
        "Show {category} products",
        "List all {category} items",
        "Find {category} equipment",
        "{category} stuff"
    ],
    "get_details": [
        "Details for {name}",
        "Tell me about {name}",
        "Info on product ID {id}",
        "What is {name}?"
    ],
    "compare_products": [
        "Compare {name1} and {name2}",
        "Differences between {name1} vs. {name2}",
        "Show {name1} vs. {name2}"
    ],
    "general_inquiry": [
        "What products do you have?",
        "List all items",
        "Show me your catalog",
        "What do you sell?"
    ],
    "out_of_scope": [
        "What's the weather?",
        "Tell me a joke",
        "How's the stock market?",
        "Who won the game?"
    ]
}

# 3. Generate synthetic query-intent dataset
data = []

# Helper function to extract keywords from description
def extract_keywords(description, n=2):
    words = description.lower().split()
    return " ".join(random.sample(words, min(n, len(words))))

# Generate queries for each product
for _, row in df_products.iterrows():
    name = row["name"]
    category = row["category"]
    description = row["description"]
    id_ = row["id"]
    description_keywords = extract_keywords(description)

    # Search Products
    for template in templates["search_products"]:
        query = template.format(
            name=name, category=category, description_keywords=description_keywords
        )
        data.append({"query": query, "intent": "search_products"})

    # Filter Category
    for template in templates["filter_category"]:
        query = template.format(category=category)
        data.append({"query": query, "intent": "filter_category"})

    # Get Details
    for template in templates["get_details"]:
        query = template.format(name=name, id=id_)
        data.append({"query": query, "intent": "get_details"})

# Compare Products (pair products randomly)
for _ in range(len(df_products) // 2):  # Generate pairs
    name1, name2 = random.sample(df_products["name"].tolist(), 2)
    for template in templates["compare_products"]:
        query = template.format(name1=name1, name2=name2)
        data.append({"query": query, "intent": "compare_products"})

# General Inquiry and Out-of-Scope (static)
for intent in ["general_inquiry", "out_of_scope"]:
    for template in templates[intent]:
        for _ in range(10):  # Repeat to balance dataset
            data.append({"query": template, "intent": intent})

# Create DataFrame and save
df_intents = pd.DataFrame(data)
df_intents.to_csv("intent_data.csv", index=False)
print(f"Generated {len(df_intents)} query-intent pairs, saved to 'intent_data.csv'")

# 4. Encode labels
label_encoder = LabelEncoder()
df_intents["label"] = label_encoder.fit_transform(df_intents["intent"])
num_labels = len(label_encoder.classes_)

# 5. Prepare dataset
dataset = Dataset.from_pandas(df_intents[["query", "label"]])
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["query"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["query"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Split into train and eval (80-20)
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 6. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir="intent_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    # evaluate_during_training=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 9. Fine-tune model
trainer.train()
trainer.save_model("intent_classifier")
tokenizer.save_pretrained("intent_classifier")
print("Fine-tuned intent classifier saved to 'intent_classifier'")

# 10. Inference example
model = AutoModelForSequenceClassification.from_pretrained("intent_classifier")
tokenizer = AutoTokenizer.from_pretrained("intent_classifier")

# query = "find a wireless mouse"
# inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
# outputs = model(**inputs)
# predicted_label = torch.argmax(outputs.logits, dim=1).item()
# predicted_intent = label_encoder.inverse_transform([predicted_label])[0]
# print(f"Query: {query}")
# print(f"Predicted Intent: {predicted_intent}")
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="intent_classifier",
    output="intent_classifier_onnx",
    task="text-classification",
    # model=model,
    tokenizer=tokenizer
)
print("Fine-tuned model exported to ONNX at 'intent_classifier_onnx'")


from optimum.onnxruntime import ORTModelForFeatureExtraction

ort_model = ORTModelForFeatureExtraction.from_pretrained("intent_classifier_onnx", provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained("intent_classifier_onnx")
print(tokenizer)

# Sample input text
from optimum.onnxruntime import ORTModelForSequenceClassification  # Changed from FeatureExtraction

model_dir = "fine_tuned_model_onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

text = "What's the weather like today?"

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