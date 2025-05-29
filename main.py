import traceback
import json
import tomllib as tomlib
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import numpy as np
from llama_index.core import (
 Settings,
 VectorStoreIndex,
 get_response_synthesizer,
 SimpleDirectoryReader,
 StorageContext,
)
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from deepeval.metrics import (
 AnswerRelevancyMetric,
 FaithfulnessMetric,
 ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from llama_index.llms.together import TogetherLLM
from neo4j import GraphDatabase

load_dotenv()


def id_func(index, document):
 """Creates a specific chunk id"""
 document_name = Path(document.metadata["file_name"]).stem
 return f"{document_name}-{index}"


def clear_neo4j(cfg):
 """
 Clears all nodes and relationships from a Neo4j database.

 Args:
     uri (str): The URI of the Neo4j database.
     user (str): The username for connecting to the database.
     password (str): The password for connecting to the database.
 """
 driver = GraphDatabase.driver(
  cfg["config"]["env"]["NEO4J_URI"],
  auth=(
   cfg["config"]["env"]["NEO4J_USER"],
   cfg["config"]["env"]["NEO4J_PASSWORD"],
  ),
 )
 with driver.session() as session:
  session.run("MATCH (n) DETACH DELETE n")
  # session.run(f"DROP INDEX `{cfg['config']['rag']['setup']['index_name']}`")
 driver.close()


PARAMS = {
 "hybrid_search": [True, False],  # to test
 # "agent": [True, False],
 "llm": ["llama-3.2-11B-vision-instruct", "gemini-1.5-pro"],
 # "embedding_provider": ["openai", "huggingface"],
 # "embedding_dimension": [128, 512, 1024],
 "embedding_dimension": 512,
 # "similarity_cutoff": [0.7, 0.8, 0.9],
 "similarity_top_k": [3, 5],
 "chunk_size": [128, 256],
 "preparation": ["TokenTextSplitter", "SentenceSplitter"],
 "chunk_overlap": 0.2,
}

number_of_experiments = 1
for key, value in PARAMS.items():
 if isinstance(value, list):
  number_of_experiments *= len(value)
print(f"{number_of_experiments=}")

CLEAR_DB = False
UPDATE_DB = False
PREDICT = False
EVALUATE = True

DATASET_SIZE = 50

i = 0
###### setup experiments
for hs in PARAMS["hybrid_search"]:
 for llm_name in PARAMS["llm"]:
  for top_k in PARAMS["similarity_top_k"]:
   for chunk_size in PARAMS["chunk_size"]:
    for preparation in PARAMS["preparation"]:
        print(
         f"{i=} {hs=}, {llm_name=}, {top_k=}, {chunk_size=}, {preparation=}"
        )
        i += 1

        ###### parse configuration
        with open("./config.toml", "rb") as file:
         cfg = tomlib.load(file)

        cfg["config"]["rag"]["setup"]["hybrid_search"] = hs
        # cfg['config']['rag']['models']['llm_openai'] = llm_name
        cfg["config"]["rag"]["setup"]["similarity_top_k"] = top_k
        cfg["config"]["rag"]["setup"]["chunk_size"] = chunk_size
        cfg["config"]["rag"]["setup"]["preparation"] = preparation

        # print(f'{cfg=}')

        ###### load data
        dataset = json.load(open("./data/q&a.json", "r"))

        base_path = Path("./data/companies")
        doc_paths = [
         path.as_posix()
         for path in base_path.iterdir()
         if path.is_file()
        ]

        ###### load models
        # initialize LLM
        if llm_name == "gemini-1.5-pro":
         llm = GoogleGenAI(
          model=llm_name,
          api_key=cfg["config"]["env"]["GEMINI_API_KEY"],
          temperature=cfg["config"]["rag"]["models"]["temperature"],
         )
        elif llm_name == "llama-3.2-11B-vision-instruct":
         llm = TogetherLLM(
          model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
         )
         # time.sleep(60)

        # initialize EM
        embed_model = OpenAIEmbedding(
         model=cfg["config"]["rag"]["models"]["em_openai"],
         api_key=cfg["config"]["env"]["OPENAI_API_KEY"],
         dimensions=cfg["config"]["rag"]["setup"]["embedding_dimension"],
        )

        Settings.llm = llm
        Settings.embed_model = embed_model

        ###### generate embeddings & fill the DB
        if CLEAR_DB:
         clear_neo4j(cfg)

        if UPDATE_DB:
         # initialize a file reader
         reader = SimpleDirectoryReader(input_files=doc_paths)

         # load documents into LlamaIndex Documents
         documents = reader.load_data()

         vector_store = Neo4jVectorStore(
          username=cfg["config"]["env"]["NEO4J_USER"],
          password=cfg["config"]["env"]["NEO4J_PASSWORD"],
          url=cfg["config"]["env"]["NEO4J_URI"],
          embedding_dimension=cfg["config"]["rag"]["setup"][
           "embedding_dimension"
          ],
          distance_strategy=cfg["config"]["rag"]["setup"][
           "distance_strategy"
          ],
          index_name=cfg["config"]["rag"]["setup"]["index_name"],
          text_node_property=cfg["config"]["rag"]["setup"][
           "text_node_property"
          ],
          hybrid_search=cfg["config"]["rag"]["setup"][
           "hybrid_search"
          ],
         )

         # setup context
         storage_context = StorageContext.from_defaults(
          vector_store=vector_store
         )

         if preparation == "TokenTextSplitter":
          parser = TokenTextSplitter(
           chunk_size=cfg["config"]["rag"]["setup"]["chunk_size"],
           chunk_overlap=int(
            np.ceil(
             cfg["config"]["rag"]["setup"]["chunk_overlap"]
             * cfg["config"]["rag"]["setup"]["chunk_size"]
            )
           ),
           separator=cfg["config"]["rag"]["setup"]["separator"],
           id_func=id_func,
          )
         elif preparation == "SentenceSplitter":
          parser = SentenceSplitter(
           chunk_size=cfg["config"]["rag"]["setup"]["chunk_size"],
           chunk_overlap=int(
            np.ceil(
             cfg["config"]["rag"]["setup"]["chunk_overlap"]
             * cfg["config"]["rag"]["setup"]["chunk_size"]
            )
           ),
           separator=cfg["config"]["rag"]["setup"]["separator"],
           id_func=id_func,
          )

         # parse documents into nodes (chunks)
         nodes = parser.get_nodes_from_documents(documents)

         db_index = VectorStoreIndex(
          nodes,
          storage_context=storage_context,
          embed_model=embed_model,
          show_progress=False,
         )
        else:
         vector_store = Neo4jVectorStore(
          username=cfg["config"]["env"]["NEO4J_USER"],
          password=cfg["config"]["env"]["NEO4J_PASSWORD"],
          url=cfg["config"]["env"]["NEO4J_URI"],
          embedding_dimension=cfg["config"]["rag"]["setup"][
           "embedding_dimension"
          ],
          distance_strategy=cfg["config"]["rag"]["setup"][
           "distance_strategy"
          ],
          index_name=cfg["config"]["rag"]["setup"]["index_name"],
          text_node_property=cfg["config"]["rag"]["setup"][
           "text_node_property"
          ],
          hybrid_search=cfg["config"]["rag"]["setup"][
           "hybrid_search"
          ],
         )

         db_index = VectorStoreIndex.from_vector_store(vector_store)

        ###### configure different tools
        # custom retriever
        retriever = VectorIndexRetriever(
         index=db_index,
         similarity_top_k=cfg["config"]["rag"]["setup"][
          "similarity_top_k"
         ],
         vector_store_query_mode=cfg["config"]["rag"]["setup"][
          "vector_store_query_mode"
         ],
        )

        # custom response synthesizer
        response_synthesizer = get_response_synthesizer(
         response_mode=cfg["config"]["rag"]["setup"]["response_mode"]
        )

        # combine custom query engine
        query_engine = RetrieverQueryEngine(
         retriever=retriever, response_synthesizer=response_synthesizer
        )

        ###### configure setup
        setup = query_engine

        ###### predict
        if PREDICT:
         predictions = []
         for item in tqdm(dataset[:DATASET_SIZE]):
          try:
           response = setup.query(item.get("question"))

           output = response.response
           context = [
            node.get_content() for node in response.source_nodes
           ]
           ids = [node.id_ for node in response.source_nodes]

           predictions.append(
            {
             "source_nodes_ids": ids,
             "source_nodes": context,
             "response": output,
            }
           )
          except Exception as e:
           print(e)
           traceback.format_exc()

         ###### save the results
         with open(
          f"./predictions/i_{i}_hs_{hs}_llm_{llm_name}_topk_{top_k}_cs_{chunk_size}_prep_{preparation}.json",
          "w",
         ) as f:
          json.dump(predictions, f)
        else:
         predictions = json.load(
          open(
           f"./predictions/i_{i}_hs_{hs}_llm_{llm_name}_topk_{top_k}_cs_{chunk_size}_prep_{preparation}.json",
           "r",
          )
         )

        ###### evaluate
        if EVALUATE:
         test_cases = []
         for j, item in enumerate(tqdm(dataset[:DATASET_SIZE])):
          # define test case
          test_case = LLMTestCase(
           input=item.get("question"),
           actual_output=predictions[j]["response"],
           expected_output=item.get("answer"),
           retrieval_context=predictions[j]["source_nodes"],
          )
          test_cases.append(test_case)

         answer_relevancy_metric = AnswerRelevancyMetric(
          model=cfg["config"]["rag"]["models"]["llm_openai"]
         )
         faithfulness_metric = FaithfulnessMetric(
          model=cfg["config"]["rag"]["models"]["llm_openai"]
         )
         contextual_relevancy_metric = ContextualRelevancyMetric(
          model=cfg["config"]["rag"]["models"]["llm_openai"]
         )

         # time.sleep(60)

         results = evaluate(
          test_cases=test_cases,
          metrics=[
           answer_relevancy_metric,
           faithfulness_metric,
           contextual_relevancy_metric,
          ],
          max_concurrent=50,
         )

         evaluations = []
         for item in results.test_results:
          evaluations.append(
           {
            "answer_relevancy": {
             "reason": item.metrics_data[0].reason,
             "score": item.metrics_data[0].score,
            },
            "faithfulness": {
             "reason": item.metrics_data[1].reason,
             "score": item.metrics_data[1].score,
            },
            "contextual_relevancy": {
             "reason": item.metrics_data[2].reason,
             "score": item.metrics_data[2].score,
            },
           }
          )

         ###### save the results
         with open(
          f"./evaluations/i_{i}_hs_{hs}_llm_{llm_name}_topk_{top_k}_cs_{chunk_size}_prep_{preparation}.json",
          "w",
         ) as f:
          json.dump(evaluations, f)
