"""Wipe all nodes/relationships from the configured Neo4j database so the
hospital_neo4j_etl service can load the new Tunisian dataset into a clean
graph, without leftover nodes from the old US-based dataset.

Reads connection details from the `.env` file in the project root (the same
keys used by hospital_neo4j_etl): NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
NEO4J_DATABASE.

Deletion is batched (10k nodes at a time) so it works reliably against
Neo4j AuraDB even for larger graphs. The 'reviews' vector index is also
dropped since the ETL recreates it (and its embedding dimension might
differ depending on which embedding model you configure).

Run with: python scripts/clear_neo4j_database.py
"""

from __future__ import annotations

import pathlib

from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _load_env(path: pathlib.Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def main() -> None:
    env = _load_env(ENV_PATH)
    uri = env.get("NEO4J_URI")
    username = env.get("NEO4J_USERNAME")
    password = env.get("NEO4J_PASSWORD")
    database = env.get("NEO4J_DATABASE", "neo4j")

    if not all([uri, username, password]):
        raise SystemExit(
            "Missing NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env "
            f"(looked in {ENV_PATH})"
        )

    print(f"Connecting to {uri} (database={database})...")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            print("Deleting all nodes and relationships in batches...")
            deleted_total = 0
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n "
                    "RETURN count(n) AS deleted"
                )
                deleted = result.single()["deleted"]
                deleted_total += deleted
                if deleted == 0:
                    break
                print(f"  deleted {deleted_total} nodes so far...")

            print("Dropping the 'reviews' vector index (if it exists)...")
            session.run("DROP INDEX reviews IF EXISTS")
    finally:
        driver.close()

    print(f"Done. Deleted {deleted_total} nodes total. Database is now empty.")


if __name__ == "__main__":
    main()
