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

### --- jaeger ----

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Configure the tracer provider
resource = Resource(attributes={
    SERVICE_NAME: "product-search-api"
})

# Create a tracer provider
tracer_provider = TracerProvider(resource=resource)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

# Add the exporter to the tracer provider
tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

# Set the tracer provider
trace.set_tracer_provider(tracer_provider)

# Get a tracer
tracer = trace.get_tracer(__name__)

# # Initialize auto-instrumentation for Flask and requests
# # Note: FlaskInstrumentor().instrument(app) should be called after app is initialized
# def instrument_flask_app(app):
#     FlaskInstrumentor().instrument_app(app)
#     RequestsInstrumentor().instrument()

### ---- end ---

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
    # Create a root span for the search products operation
    from opentelemetry.context import set_value, get_current, attach, detach
    # # Start the parent span
    # span = tracer.start_span("search_products_operation2")
    # span.set_attribute("endpoint", "search-products")

    # # Set the span as the current span in a new context
    # context = trace.set_span_in_context(span, context=get_current())
    # token = attach(context)  # Attach the context

    # # Start the child span (inherits parent from current context)
    # child_span = tracer.start_span("query_find_intent2")
    # # Do stuff...
    # child_span.end()

    # find_in_mongo_span = tracer.start_span("find_in_mongo")
    # find_in_mongo_span.end()
    # # Clean up
    # span.end()
    # detach(token)  # Detach the context
        
    # Start the parent span
    span = tracer.start_span("search_products_operation2")
    span.set_attribute("endpoint", "search-products")

    # Set the parent span as the current span in a new context
    context = trace.set_span_in_context(span, context=get_current())
    token = attach(context)  # Attach the context

    # Start the child span (inherits parent from current context)
    child_span = tracer.start_span("query_find_intent2")
    child_span.set_attribute("operation", "find_intent")

    # Set the child span as the current span in a new context
    child_context = trace.set_span_in_context(child_span, context=get_current())
    child_token = attach(child_context)  # Attach the child context

    # Start the grandchild span (inherits child as parent from current context)
    grandchild_span = tracer.start_span("grandchild_operation")
    grandchild_span.set_attribute("sub_operation", "process_data")

    # Do stuff in grandchild span...
    grandchild_span.end()

    # Detach the child context to restore the parent context
    detach(child_token)

    # End the child span
    child_span.end()

    # Start the find_in_mongo span (inherits parent from current context)
    find_in_mongo_span = tracer.start_span("find_in_mongo")
    find_in_mongo_span.set_attribute("db_operation", "mongo_query")

    # Do stuff in find_in_mongo span...
    find_in_mongo_span.end()

    # End the parent span and detach the parent context
    span.end()
    detach(token)  # Detach the context    
        
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
            
            # # Replace Qdrant results with complete MongoDB documents
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
        # "mongo_products": mongo_products,
        "q": text,
        "predicted_class": predicted_class
    }), 200


app.run(debug=True, host='0.0.0.0', port=5001)
