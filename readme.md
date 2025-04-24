
# todo

- docker compose with ~> Quadrent-db, Apache-airflow, Mongo-db
- generate data
- fix llm model and dry run idea 
- build first DAG in Airflow (1: take products and store embeddings, 2: embed new products)
- search API 



# Phases

## initial  
- I have 5000 (or N) products data
- fine tune the embedding model (M1)
- fine tune the intent identification model (M2)
- need to store them in Qdrant after generating embedding with llm model

## daily updates 
- process daily new incoming products and update embedding or whatever is needed to improve relevance  
-

## build search api
- incoming requests are passed to M1 to identify intent
- based to intent search quadrent to fetch relevent items / products
- show response 

## logging and tracing  (monitoring)
- add trace on user request on search products
  T1 ~> (req -> M1 -> M1 res -> Qdrant res-> mongo /elastic /db -> response ) ; L1 (total time taken logged )
- plot T1

## alert 
- if L1 is high


## text

will be given 5k (on any data) 
will have to 
1. fine tune embedding model with it (A1)
2. fine tune intent model with it (A2)
use the embedding model to insert relevant data in Qdrant (A3)
build api to fetch data from query
 - identify query intent
 - do search based on intent and fetch results
 - respond with results 

 extra implement logging, tracing, alert 


 agent for multi modal 
 guard rail 
 scale ml model