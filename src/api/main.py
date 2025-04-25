from flask import Flask, jsonify, request

##  --------------- ml code --------------------

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification  # Changed from FeatureExtraction
import torch

model_dir = "intent_classifier_onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# ----------------------------------------------


## ---- Qdrant --- 
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initialize Qdrant client
qdrant = QdrantClient("http://localhost:6333")

# Initialize the sentence transformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Collection name for product embeddings
collection_name = "product_embeddings"

# Function to search products using vector similarity
def search_products_in_qdrant(query_text, limit=5):
    # Generate embedding for the query
    query_embedding = model.encode(query_text)
    
    # Search for similar products in Qdrant
    search_results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    
    # Extract product information from search results
    products = [result.payload for result in search_results]
    return products

### ----------- end ----


### ---- connect to mongo
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["hack"]  # Database name
products_collection = db["products"]  # Collection name



### ---- end ----


### ---- redis --- 

### ---------------

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """
    Base endpoint that returns a 200 status code.
    """
    return jsonify({
        "status": "success",
        "message": "API is running"
    }), 200

@app.route('/search-products', methods=['GET'])
def search_products():
    """
    Endpoint to search for products.
    Receives query parameter 'q' for search text.
    Currently returns an empty products array with a 200 status code.
    """
    text = request.args.get('q', '')
    print({text})
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Convert to numpy arrays for ONNX Runtime
    onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

    # Get predictions
    outputs = ort_model(**onnx_inputs)
    logits = outputs.logits

    # Get predicted class (intent)
    predicted_class = torch.argmax(torch.from_numpy(logits), dim=1).item()

    print(f"Predicted intent class: {predicted_class}")
    
    
    # Map predicted class to intent
    intent_mapping = {
        0: "search_products",
        1: "filter_category",
        2: "get_details",
        3: "compare_products",
        4: "general_inquiry",
        5: "out_of_scope"
    }
    
    intent = intent_mapping.get(predicted_class, "unknown")
    print(f"Mapped intent: {intent}")
    
    # Only search in Qdrant if the intent is related to products
    products = []

    try:
        # Load the embedding model
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embedding for the query
        query_embedding = model.encode([text])
        
        # Search in Qdrant
        collection_name = "product_embeddings"

        print(' --- getting results from qqdrant ----')
        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=5  # Return top 5 results
        )
        print(' ----- got results from qdrant ---')
        # Extract product information from search results
        products = [result.payload for result in search_results]
        print(f"Found {len(products)} products matching the query")
        
        # query_text = text #"mouse"
        # print('> query qdrant {}',query_text)
        # query_embedding = model.encode([query_text])[0]

        # print('> searching results')
        # search_result = qdrant.search(
        #     collection_name=collection_name,
        #     query_vector=query_embedding,
        #     limit=3  # Top 3 similar products
        # )

        # # Print results
        # print("\nSearch Results:")
        # for hit in search_result:
        #     print(f"Product: {hit.payload['name']}, Category: {hit.payload['category']}, Score: {hit.score}")
        #     print(hit.payload)
        #     products.append({
        #         'name': hit.payload['name']
        #     })
        # # products = [result.payload for result in search_results]
        
    except Exception as e:
        print(f"Error searching Qdrant: {str(e)}")
    # Search MongoDB for products with matching IDs
    mongo_products = []
    try:
        from bson.objectid import ObjectId
        from pymongo import MongoClient
        

        
        # Extract product IDs from Qdrant results
        product_ids = [product.get("id") for product in products if product.get("id")]
        
        # Convert string IDs to ObjectId for MongoDB query
        object_ids = [ObjectId(pid) for pid in product_ids if pid]
        
        if object_ids:
            # Query MongoDB for the complete product information
            mongo_products = list(products_collection.find({"_id": {"$in": object_ids}}))
            
            # Replace Qdrant results with complete MongoDB documents
            # if mongo_products:
            #     # Convert MongoDB _id to string for JSON serialization
            #     for product in mongo_products:
            #         if "_id" in product:
            #             product["_id"] = str(product["_id"])
                
            #     products = mongo_products
            #     print(f"Retrieved {len(products)} complete products from MongoDB")
            # else:
            #     print("No matching products found in MongoDB")
        else:
            print("No valid product IDs to search in MongoDB")
            
    except Exception as e:
        print(f"Error searching MongoDB: {str(e)}")
    ## search the vector result ids in mongo
    return jsonify({
        "status": "success",
        "products": products,
        "mongo_products": mongo_products,
        "q": text,
        "predicted_class": predicted_class
    }), 200


app.run(debug=True, host='0.0.0.0', port=5001)
