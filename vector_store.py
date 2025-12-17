import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import chromadb


LOGGER = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, storage_path: Path, collection_name: str) -> None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(storage_path))
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=None)
        LOGGER.info("Connected to Chroma collection=%s at %s", collection_name, storage_path)

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        self.collection.upsert(
            ids=list(ids),
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=documents,
        )

    def query(self, query_embedding: np.ndarray, top_k: int) -> chromadb.api.models.Collection.QueryResult:
        try:
            return self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except ValueError:
            # fallback: minimal include
            return self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "documents"],
            )

    def count(self) -> int:
        return int(self.collection.count())

    def delete(self, ids: Sequence[str]) -> None:
        self.collection.delete(ids=list(ids))

    def reset(self) -> None:
        # 删除整个 collection 并重建
        name = self.collection.name
        self.client.delete_collection(name=name)
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=None)

    def get_all_ids_and_meta(self) -> List[tuple]:
        # 用于 remove_paper：取出所有 id + metadata（小规模作业足够用）
        data = self.collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        return list(zip(ids, metas))
