from flask import Flask, jsonify, request, Response
import time


## URLS 

QDRANT_URL = "http://localhost:6333"
MONGO_URL = "mongodb://localhost:27017/"
##  --------------- ml code --------------------

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification  # Changed from FeatureExtraction
import torch

# model_dir = "intent_classifier_onnx"
model_dir = "out_models/intent_classifier_onnx"
ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, provider="CPUExecutionProvider")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# ----------------------------------------------


## ---- Qdrant --- 
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Initialize Qdrant client
qdrant = QdrantClient(QDRANT_URL)
# qdrant = QdrantClient("http://qdrant:6333")

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
client = MongoClient(MONGO_URL)
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


### --- promethuse ---
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
# Initialize Prometheus histogram for search_products latency
SEARCH_LATENCY = Histogram(
    "search_products_latency_seconds",
    "Latency of the search_products endpoint",
    labelnames=["endpoint", "intent"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)  # Fine-grained buckets
)


### ----------------
### ---- redis --- 
import redis
import json
from datetime import timedelta

# Configure Redis connection
redis_client = redis.Redis(
    host='localhost',  # Use 'redis' when running in Docker
    port=6379,
    password='admin',  # Password as specified in docker-compose
    username='admin',  # Username as specified in docker-compose
    decode_responses=True  # Automatically decode responses to strings
)

# Redis utility functions
def get_from_redis(key):
    """
    Get a value from Redis by key.
    
    Args:
        key (str): The Redis key to retrieve
        
    Returns:
        The value if found, None otherwise
    """
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        print(f"Error retrieving from Redis: {str(e)}")
        return None

def store_in_redis(key, value, ttl_seconds=3):
    """
    Store a value in Redis with a specified TTL.
    
    Args:
        key (str): The Redis key
        value (any): The value to store (will be JSON serialized)
        ttl_seconds (int): Time-to-live in seconds, defaults to 3
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        serialized_value = json.dumps(value)
        redis_client.setex(key, timedelta(seconds=ttl_seconds), serialized_value)
        return True
    except Exception as e:
        print(f"Error storing in Redis: {str(e)}")
        return False


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

@app.route("/metrics", methods=["GET"])
def metrics():
    # Serve Prometheus metrics
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/search-products', methods=['GET'])
def search_products():
    """
    Endpoint to search for products.
    Receives query parameter 'q' for search text.
    Currently returns an empty products array with a 200 status code.
    """
    
    start_time = time.time()
    text = request.args.get('q', '')
    
    # Check if we have cached results for this query
    cache_key = f"search:{text}"
    cached_result = get_from_redis(cache_key)
    if cached_result:
        print(f"Cache hit for query: {text}")
        return jsonify(cached_result), 200
    
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
    span = tracer.start_span("search_products_operation")
    span.set_attribute("endpoint", "search-products")

    # Set the parent span as the current span in a new context
    context = trace.set_span_in_context(span, context=get_current())
    token = attach(context)  # Attach the context

    # Start the child span (inherits parent from current context)
    # child_span = tracer.start_span("query_find_intent")
    # child_span.set_attribute("operation", "find_intent")

    # Set the child span as the current span in a new context
    # child_context = trace.set_span_in_context(child_span, context=get_current())
    # child_token = attach(child_context)  # Attach the child context

    # # Start the grandchild span (inherits child as parent from current context)
    # grandchild_span = tracer.start_span("grandchild_operation")
    # grandchild_span.set_attribute("sub_operation", "process_data")

    # # Do stuff in grandchild span...
    # grandchild_span.end()

    # Detach the child context to restore the parent context
    # detach(child_token)

    # End the child span
    # child_span.end()

 
 
        
    print({text})
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Convert to numpy arrays for ONNX Runtime
    onnx_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

    # Get predictions
    outputs = ort_model(**onnx_inputs)
    logits = outputs.logits

    # Get predicted class (intent)
    intent_child_span = tracer.start_span("query_find_intent")
    intent_child_span.set_attribute("operation", "find_intent")
   
    predicted_class = torch.argmax(torch.from_numpy(logits), dim=1).item()
    intent_child_span.end()

    
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
        search_quad_child_span = tracer.start_span(" search_quad")
        search_quad_child_span.set_attribute("operation", " search_quad")
        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=5  # Return top 5 results
        )
        search_quad_child_span.end()
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
       # Start the find_in_mongo span (inherits parent from current context)
    find_in_mongo_span = tracer.start_span("find_in_mongo")
    find_in_mongo_span.set_attribute("db_operation", "mongo_query")


    try:
        from bson.objectid import ObjectId
        from pymongo import MongoClient
        

        
        # Extract product IDs from Qdrant results
        product_ids = [product.get("id") for product in products if product.get("id")]
        print(product_ids)
        # Convert string IDs to ObjectId for MongoDB query
        object_ids = [ObjectId(pid) for pid in product_ids if pid]
        print(object_ids)
        if object_ids:
            print("finding mongo products")
            # Query MongoDB for the complete product information
            mongo_products = list(products_collection.find({"_id": {"$in": object_ids}}, {"_id": 0}))
            
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
    
    # Do stuff in find_in_mongo span...
    find_in_mongo_span.end()
    ## search the vector result ids in mongo
    latency = time.time() - start_time
    SEARCH_LATENCY.labels(endpoint="search_products", intent=intent).observe(latency)
    # Get the current trace ID if available

    trace_id = span.get_span_context().trace_id
    print(f"Trace ID: {format(trace_id, '032x')}")
    
    # Prepare response
    response = {
        "status": "success",
        "products": mongo_products, #products,
        # "mongo_products": mongo_products,
        "q": text,
        "predicted_class": predicted_class,
        "product_ids": product_ids,
        "trace_id": format(trace_id, '032x')
    }
    
    # Cache the response
    store_in_redis(cache_key, response, ttl_seconds=60)  # Cache for 60 seconds
    # End the parent span and detach the parent context
    span.end()
    detach(token)  # Detach the context           
    return jsonify(response), 200


import signal
import sys

def graceful_shutdown(signum, frame):
    print("Received shutdown signal. Shutting down gracefully...")
    # Close any open resources, connections, etc.
    # Close MongoDB connection if it's open
    if 'client' in globals():
        client.close()
        print("MongoDB connection closed")
    # Close any other resources
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)  # Ctrl+C
signal.signal(signal.SIGTERM, graceful_shutdown)  # Termination signal

if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Error running the application: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code that runs regardless of how the app exits
        if 'client' in globals():
            client.close()
            print("MongoDB connection closed")
