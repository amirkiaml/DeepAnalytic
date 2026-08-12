"""
vectorstore.py

Non-interactive Pinecone index management, driven by config.py.
"""

import time
from pinecone import Pinecone, ServerlessSpec
from config import settings


class VectorDB:
    def __init__(self, cloud="aws", region="us-east-1"):
        # us-east-1 — free-tier-supported region (us-west-2 requires a paid plan)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.spec = ServerlessSpec(cloud=cloud, region=region)
        self.index_name = None
        self.index = None

    def list_indexes(self):
        return self.pc.list_indexes()

    def create_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        self.index_name = index_name
        existing = [idx["name"] for idx in self.list_indexes()]

        if index_name not in existing:
            print(f"Creating index '{index_name}' (dim={dimension}, metric={metric})...")
            self.pc.create_index(
                index_name,
                dimension=dimension,
                metric=metric,
                spec=self.spec,
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print(f"Index '{index_name}' created and ready.")
        else:
            print(f"Index '{index_name}' already exists — connecting to it as-is.")

    def connect_to_index(self, index_name: str = None):
        index_name = index_name or self.index_name
        existing = [idx["name"] for idx in self.list_indexes()]
        if index_name not in existing:
            raise Exception(f"Index '{index_name}' does not exist.")
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        return self.index

    def delete_index(self, index_name: str):
        existing = [idx["name"] for idx in self.list_indexes()]
        if index_name in existing:
            self.pc.delete_index(index_name)
            print(f"Deleted index '{index_name}'.")
        else:
            print(f"Index '{index_name}' does not exist — nothing to delete.")
