import logging
import os

import requests
from neo4j import GraphDatabase
from retry import retry

HOSPITALS_CSV_PATH = os.getenv("HOSPITALS_CSV_PATH")
PAYERS_CSV_PATH = os.getenv("PAYERS_CSV_PATH")
PHYSICIANS_CSV_PATH = os.getenv("PHYSICIANS_CSV_PATH")
PATIENTS_CSV_PATH = os.getenv("PATIENTS_CSV_PATH")
VISITS_CSV_PATH = os.getenv("VISITS_CSV_PATH")
REVIEWS_CSV_PATH = os.getenv("REVIEWS_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HOSPITAL_EMBEDDING_MODEL = os.getenv(
    "HOSPITAL_EMBEDDING_MODEL", "text-embedding-ada-002"
)
REVIEW_TEXT_PROPERTIES = [
    "physician_name",
    "patient_name",
    "text",
    "hospital_name",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


NODES = ["Hospital", "Payer", "Physician", "Patient", "Visit", "Review"]


def _embed_texts(texts):
    """Embed a batch of texts via the configured OpenAI-compatible
    embeddings endpoint (NVIDIA NIM if NVIDIA_API_KEY is set, otherwise
    plain OpenAI). Called directly via HTTP rather than langchain, so we
    control exactly what gets sent (plain text, no legacy Neo4j
    procedures involved).
    """
    if NVIDIA_API_KEY:
        api_key = NVIDIA_API_KEY
        base_url = NVIDIA_BASE_URL
        extra_body = {"input_type": "passage"}
    else:
        api_key = OPENAI_API_KEY
        base_url = "https://api.openai.com/v1"
        extra_body = {}

    payload = {"model": HOSPITAL_EMBEDDING_MODEL, "input": texts, **extra_body}
    response = requests.post(
        f"{base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def _set_uniqueness_constraints(tx, node):
    query = f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node})
        REQUIRE n.id IS UNIQUE;"""
    _ = tx.run(query, {})


@retry(tries=100, delay=10)
def load_hospital_graph_from_csv() -> None:
    """Load structured hospital CSV data following
    a specific ontology into Neo4j"""

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    LOGGER.info("Setting uniqueness constraints on nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        for node in NODES:
            session.execute_write(_set_uniqueness_constraints, node)

    LOGGER.info("Loading hospital nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{HOSPITALS_CSV_PATH}' AS hospitals
        MERGE (h:Hospital {{id: toInteger(hospitals.hospital_id),
                            name: hospitals.hospital_name,
                            governorate: hospitals.hospital_state}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading payer nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PAYERS_CSV_PATH}' AS payers
        MERGE (p:Payer {{id: toInteger(payers.payer_id),
        name: payers.payer_name}});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading physician nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PHYSICIANS_CSV_PATH}' AS physicians
        MERGE (p:Physician {{id: toInteger(physicians.physician_id),
                            name: physicians.physician_name,
                            dob: physicians.physician_dob,
                            grad_year: physicians.physician_grad_year,
                            school: physicians.medical_school,
                            salary: toFloat(physicians.salary)
                            }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading visit nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
        MERGE (v:Visit {{id: toInteger(visits.visit_id),
                            room_number: toInteger(visits.room_number),
                            admission_type: visits.admission_type,
                            admission_date: visits.date_of_admission,
                            test_results: visits.test_results,
                            status: visits.visit_status
        }})
            ON CREATE SET v.chief_complaint = visits.chief_complaint
            ON MATCH SET v.chief_complaint = visits.chief_complaint
            ON CREATE SET v.treatment_description = visits.treatment_description
            ON MATCH SET v.treatment_description = visits.treatment_description
            ON CREATE SET v.diagnosis = visits.primary_diagnosis
            ON MATCH SET v.diagnosis = visits.primary_diagnosis
            ON CREATE SET v.discharge_date = visits.discharge_date
            ON MATCH SET v.discharge_date = visits.discharge_date
         """
        _ = session.run(query, {})

    LOGGER.info("Loading patient nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{PATIENTS_CSV_PATH}' AS patients
        MERGE (p:Patient {{id: toInteger(patients.patient_id),
                        name: patients.patient_name,
                        sex: patients.patient_sex,
                        dob: patients.patient_dob,
                        blood_type: patients.patient_blood_type
                        }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading review nodes")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS
        FROM '{REVIEWS_CSV_PATH}' AS reviews
        MERGE (r:Review {{id: toInteger(reviews.review_id),
                         text: reviews.review,
                         patient_name: reviews.patient_name,
                         physician_name: reviews.physician_name,
                         hospital_name: reviews.hospital_name
                        }});
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'AT' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS row
        MATCH (source: `Visit` {{ `id`: toInteger(trim(row.`visit_id`)) }})
        MATCH (target: `Hospital` {{ `id`: toInteger(trim(row.`hospital_id`))}})
        MERGE (source)-[r: `AT`]->(target)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'WRITES' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{REVIEWS_CSV_PATH}' AS reviews
            MATCH (v:Visit {{id: toInteger(reviews.visit_id)}})
            MATCH (r:Review {{id: toInteger(reviews.review_id)}})
            MERGE (v)-[writes:WRITES]->(r)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'TREATS' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[treats:TREATS]->(v)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'COVERED_BY' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MATCH (p:Payer {{id: toInteger(visits.payer_id)}})
            MERGE (v)-[covered_by:COVERED_BY]->(p)
            ON CREATE SET
                covered_by.service_date = visits.discharge_date,
                covered_by.billing_amount = toFloat(visits.billing_amount)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'HAS' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (p:Patient {{id: toInteger(visits.patient_id)}})
            MATCH (v:Visit {{id: toInteger(visits.visit_id)}})
            MERGE (p)-[has:HAS]->(v)
        """
        _ = session.run(query, {})

    LOGGER.info("Loading 'EMPLOYS' relationships")
    with driver.session(database=NEO4J_DATABASE) as session:
        query = f"""
        LOAD CSV WITH HEADERS FROM '{VISITS_CSV_PATH}' AS visits
            MATCH (h:Hospital {{id: toInteger(visits.hospital_id)}})
            MATCH (p:Physician {{id: toInteger(visits.physician_id)}})
            MERGE (h)-[employs:EMPLOYS]->(p)
        """
        _ = session.run(query, {})

    LOGGER.info("Creating vector index for reviews")
    with driver.session(database=NEO4J_DATABASE) as session:
        # Created here (using modern Cypher syntax) rather than left to
        # langchain's Neo4jVector, which only knows the legacy
        # `db.index.vector.createNodeIndex` procedure that newer Neo4j
        # versions have removed. langchain will detect this index already
        # exists and skip its own (broken) creation step.
        query = (
            "CREATE VECTOR INDEX reviews IF NOT EXISTS "
            "FOR (r:Review) ON (r.embedding) "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: $dimensions, "
            "`vector.similarity_function`: 'cosine' "
            "} }"
        )
        _ = session.run(query, {"dimensions": EMBEDDING_DIMENSION})

    LOGGER.info("Populating review embeddings")
    # Done here via plain `SET` rather than left to langchain's
    # Neo4jVector, which uses the legacy `db.create.setVectorProperty`
    # procedure that newer Neo4j versions have removed (modern Neo4j
    # supports storing float-array properties directly).
    # Kept small on purpose: writing many large embedding vectors
    # (1024+ floats each) in a single transaction can produce a payload
    # large enough to trip connection/write failures on some networks.
    # Smaller batches also mean progress persists across ETL retries,
    # since already-embedded reviews are skipped by the WHERE clause
    # below.
    EMBEDDING_BATCH_SIZE = 50

    @retry(tries=5, delay=5)
    def _write_embedding_batch(session, rows):
        write_query = (
            "UNWIND $rows AS row "
            "MATCH (n:Review) WHERE elementId(n) = row.id "
            "SET n.embedding = row.embedding"
        )
        session.run(write_query, {"rows": rows})

    with driver.session(database=NEO4J_DATABASE) as session:
        while True:
            fetch_query = (
                "MATCH (n:Review) WHERE n.embedding IS NULL "
                "RETURN elementId(n) AS id, reduce(str='', "
                "k IN $props | str + '\\n' + k + ':' + "
                "coalesce(n[k], '')) AS text "
                "LIMIT $limit"
            )
            batch = session.run(
                fetch_query,
                {"props": REVIEW_TEXT_PROPERTIES, "limit": EMBEDDING_BATCH_SIZE},
            ).data()
            if not batch:
                break

            embeddings = _embed_texts([row["text"] for row in batch])

            rows = [
                {"id": row["id"], "embedding": embedding}
                for row, embedding in zip(batch, embeddings)
            ]
            _write_embedding_batch(session, rows)

            if len(batch) < EMBEDDING_BATCH_SIZE:
                break


if __name__ == "__main__":
    load_hospital_graph_from_csv()
