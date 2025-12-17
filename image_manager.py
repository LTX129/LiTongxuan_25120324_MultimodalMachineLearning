import logging
from pathlib import Path
from typing import Dict, List

from embeddings import EmbeddingManager
from vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


class ImageManager:
    def __init__(self, embedding_manager: EmbeddingManager, store: VectorStore) -> None:
        self.embedding_manager = embedding_manager
        self.store = store

    def index_folder(self, folder: Path) -> List[Path]:
        folder = folder.expanduser().resolve()
        images = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        if not images:
            LOGGER.info("No images found in %s", folder)
            return []
        embeddings = self.embedding_manager.embed_images(images)
        import hashlib

        def _file_id(p: Path) -> str:
            h = hashlib.sha1()
            stat = p.stat()
            h.update(str(p.resolve()).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
            return h.hexdigest()[:20]

        ids = [_file_id(p) for p in images]
        metadatas = [{"path": str(p)} for p in images]
        captions = [p.name for p in images]
        self.store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=captions)
        LOGGER.info("Indexed %d images from %s", len(images), folder)
        return images

    def search_by_text(self, query: str, top_k: int) -> List[Dict[str, str]]:
        query_embedding = self.embedding_manager.embed_clip_text([query]).squeeze(0)
        raw = self.store.query(query_embedding, top_k)
        results: List[Dict[str, str]] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        for idx, caption in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            results.append(
                {
                    "id": ids[idx] if idx < len(ids) else "",
                    "caption": caption,
                    "path": meta.get("path", ""),
                }
            )
        return results
