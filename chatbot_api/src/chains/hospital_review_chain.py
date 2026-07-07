import os

from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI

from utils.compatible_embeddings import SimpleOpenAICompatibleEmbeddings
from utils.llm_config import embedding_provider_kwargs, llm_provider_kwargs

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")
HOSPITAL_EMBEDDING_MODEL = os.getenv(
    "HOSPITAL_EMBEDDING_MODEL", "text-embedding-ada-002"
)

# The vector index and review embeddings are created and populated by the
# hospital_neo4j_etl service (using modern Neo4j syntax), so we connect to
# the existing index here rather than using `from_existing_graph`, which
# relies on legacy procedures (`db.index.vector.createNodeIndex`,
# `db.create.setVectorProperty`) that newer Neo4j versions have removed.
REVIEW_TEXT_PROPERTIES = [
    "physician_name",
    "patient_name",
    "text",
    "hospital_name",
]
REVIEW_RETRIEVAL_QUERY = (
    f"RETURN reduce(str='', k IN {REVIEW_TEXT_PROPERTIES} |"
    " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
    "node {.*, `embedding`: Null, id: Null, "
    + ", ".join([f"`{prop}`: Null" for prop in REVIEW_TEXT_PROPERTIES])
    + "} AS metadata, score"
)

neo4j_vector_index = Neo4jVector.from_existing_index(
    embedding=SimpleOpenAICompatibleEmbeddings(
        model=HOSPITAL_EMBEDDING_MODEL, **embedding_provider_kwargs()
    ),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
    index_name="reviews",
    node_label="Review",
    embedding_node_property="embedding",
    retrieval_query=REVIEW_RETRIEVAL_QUERY,
)

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0, **llm_provider_kwargs()),
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)
reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt
