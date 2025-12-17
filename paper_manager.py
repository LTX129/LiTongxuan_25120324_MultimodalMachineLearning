import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from collections import defaultdict

import config
from embeddings import EmbeddingManager
from pdf_utils import extract_text_chunks
from vector_store import VectorStore
from text_filters import is_reference_like


LOGGER = logging.getLogger(__name__)


class PaperManager:
    def __init__(self, embedding_manager: EmbeddingManager, store: VectorStore) -> None:
        self.embedding_manager = embedding_manager
        self.store = store
        # ---- 新增：topic embedding 缓存 ----
        self._topic_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    def organize_folder(self, folder: Path, topics: str) -> List[Dict[str, str]]:
        folder = folder.expanduser().resolve()
        pdfs = list(folder.rglob("*.pdf"))
        results: List[Dict[str, str]] = []
        if not pdfs:
            LOGGER.info("No PDFs found in %s", folder)
            return results
        for pdf in tqdm(pdfs, desc="Organizing papers"):
            try:
                info = self.add_paper(pdf, topics)
                results.append(info)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Failed to process %s: %s", pdf, exc)
        return results

    def index_existing(self, folder: Path) -> List[Dict[str, str]]:
        folder = folder.expanduser().resolve()
        pdfs = list(folder.rglob("*.pdf"))
        results: List[Dict[str, str]] = []
        if not pdfs:
            LOGGER.info("No PDFs found in %s", folder)
            return results
        for pdf in tqdm(pdfs, desc="Indexing papers"):
            try:
                info = self.add_paper(pdf, topics=None)
                results.append(info)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Failed to index %s: %s", pdf, exc)
        return results

    def search(self, query: str, top_k: int) -> List[Dict[str, str]]:
        query_embedding = self.embedding_manager.embed_text([query]).squeeze(0)

        # ✅ 先多取一些候选，再过滤 refs
        fetch_k = max(top_k * int(getattr(config, "SEARCH_FETCH_MULTIPLIER", 6)), top_k)
        raw = self.store.query(query_embedding, fetch_k)

        results: List[Dict[str, str]] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        filter_refs = bool(getattr(config, "FILTER_REFERENCE_CHUNKS", True))

        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            is_ref = (meta.get("is_ref", "0") == "1")

            # 默认过滤参考文献 chunk
            if filter_refs and is_ref:
                continue

            results.append(
                {
                    "chunk": doc,
                    "source": meta.get("source", ""),
                    "topic": meta.get("topic", ""),
                    "chunk_idx": meta.get("chunk_idx", ""),
                    "id": ids[idx] if idx < len(ids) else "",
                }
            )

            if len(results) >= top_k:
                break

        return results

    def search_grouped(self, query: str, top_k: int) -> List[Dict]:
        def _distance_to_score(dist: float) -> float:
            # 常见情况：cosine distance = 1 - cosine_sim
            # 转成“越大越好”的相似度近似
            try:
                return 1.0 - float(dist)
            except Exception:
                return 0.0

        query_embedding = self.embedding_manager.embed_text([query]).squeeze(0)

        fetch_k = max(
            top_k * int(getattr(config, "SEARCH_FETCH_MULTIPLIER", 8)),
            top_k
        )
        raw = self.store.query(query_embedding, fetch_k)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0] if "distances" in raw else [1.0] * len(documents)

        # 1) chunk-level 结果按 source 分组
        grouped = defaultdict(list)
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            source = meta.get("source", "")
            topic = meta.get("topic", "")
            chunk_idx = meta.get("chunk_idx", "")
            dist = distances[i] if i < len(distances) else 1.0
            score = _distance_to_score(dist)

            grouped[source].append({
                "score": score,
                "chunk": doc,
                "chunk_idx": chunk_idx,
                "topic": topic,
                "id": ids[i] if i < len(ids) else "",
            })

        # 2) 每篇论文：挑 top-snippets 个片段
        snippets_per_paper = int(getattr(config, "SNIPPETS_PER_PAPER", 2))
        paper_results: List[Dict] = []
        for source, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            best = items_sorted[0]
            top_snips = items_sorted[:max(1, snippets_per_paper)]
            paper_results.append({
                "source": source,
                "topic": best.get("topic", ""),
                "best_score": best.get("score", 0.0),
                "snippets": [
                    {
                        "score": it["score"],
                        "chunk_idx": it.get("chunk_idx", ""),
                        "text": it["chunk"],
                    }
                    for it in top_snips
                ],
            })

        # 3) 论文按最佳分数排序，返回 top_k 篇
        paper_results.sort(key=lambda x: x["best_score"], reverse=True)
        return paper_results[:top_k]

    def _canonical_path(self, pdf_path: Path) -> Path:
        pdf_path = pdf_path.expanduser().resolve()
        config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        if config.LIBRARY_DIR in pdf_path.parents:
            return pdf_path
        dest = config.LIBRARY_DIR / pdf_path.name
        if dest != pdf_path:
            if dest.exists():
                dest.unlink()
            shutil.copy2(pdf_path, dest)
            LOGGER.info("Copied into library %s -> %s", pdf_path, dest)
        return dest

    # ---- 替换：把 chunk_idx 写进 metadata，后续可展示更清晰 ----
    def _index_chunks(self, pdf_path: Path, chunks: List[str], topic: Optional[str]) -> None:
        embeddings = self.embedding_manager.embed_text(chunks)
        base_id = pdf_path.stem
        ids: List[str] = []
        metadatas: List[Dict[str, str]] = []

        for idx, chunk in enumerate(chunks):
            ids.append(f"{base_id}-{uuid.uuid4().hex[:8]}-{idx}")
            metadatas.append(
                {
                    "source": str(pdf_path),
                    "chunk_idx": str(idx),
                    "topic": topic or "unknown",
                    "is_ref": "1" if is_reference_like(chunk) else "0",  # ✅ 新增
                }
            )
        self.store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)

    # ---- 新增：topic 描述文本 ----
    def _topic_text(self, topic: str) -> str:
        t = topic.strip()
        if not t:
            return ""
        # 优先用配置里的描述
        desc = getattr(config, "TOPIC_DESC", {}).get(t, None)
        if desc:
            return f"Topic: {t}. Description: {desc}"
        # fallback：至少把缩写扩展成“Topic: xxx”
        return f"Topic: {t}"

    # ---- 替换：分类=top-m chunks 投票/聚合 + topic embedding 缓存 ----
    def _classify_topic(self, chunks_for_classify: Sequence[str], topics: Sequence[str]) -> Tuple[str, float]:
        topic_list = [t.strip() for t in topics if t.strip()]
        if not topic_list:
            return "uncategorized", 0.0

        # 1) embed chunks（分类用少量 chunks：前几页、截断 references）
        chunk_vecs = self.embedding_manager.embed_text(chunks_for_classify)  # (C, D)
        if chunk_vecs.size == 0:
            return "uncategorized", 0.0

        # 2) embed topics with descriptions + cache
        cache_key = tuple(topic_list)
        if cache_key in self._topic_cache:
            topic_vecs = self._topic_cache[cache_key]
        else:
            topic_texts = [self._topic_text(t) for t in topic_list]
            topic_vecs = self.embedding_manager.embed_text(topic_texts)  # (T, D)
            self._topic_cache[cache_key] = topic_vecs

        # 3) similarity matrix: (T, C) = topic_vecs @ chunk_vecs.T
        sims = topic_vecs @ chunk_vecs.T

        # 4) aggregate per topic by top-m chunks
        m = int(getattr(config, "CLASSIFY_TOP_M_CHUNKS", 6))
        m = max(1, min(m, sims.shape[1]))
        topm = np.sort(sims, axis=1)[:, -m:]  # (T, m)
        scores = topm.mean(axis=1)  # (T,)

        best_idx = int(np.argmax(scores))
        return topic_list[best_idx], float(scores[best_idx])

    # ---- 替换：分类用“前N页+references截断”，索引用“全篇” ----
    def add_paper(self, pdf_path: Path, topics: Optional[str]) -> Dict[str, str]:
        target_path = self._canonical_path(pdf_path)

        # A) 分类 chunks（更干净）
        classify_chunks = extract_text_chunks(
            target_path,
            chunk_size=config.PDF_CHUNK_SIZE,
            max_pages=getattr(config, "PDF_CLASSIFY_MAX_PAGES", 5),
            stop_at_references=getattr(config, "PDF_STOP_AT_REFERENCES", True),
        )
        # B) 索引 chunks（尽量全）
        index_chunks = extract_text_chunks(
            target_path,
            chunk_size=config.PDF_CHUNK_SIZE,
            max_pages=None,
            stop_at_references=False,
        )
        if not index_chunks:
            raise ValueError(f"No text found in {target_path}")

        topic = None
        score = None

        if topics:
            topic, score = self._classify_topic(classify_chunks or index_chunks[:3], topics.split(","))

            # 归档：library/<topic>/xxx.pdf
            dest_dir = config.LIBRARY_DIR / topic
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / target_path.name

            if dest != target_path:
                if dest.exists():
                    dest.unlink()
                shutil.move(str(target_path), str(dest))
                LOGGER.info("Classified %s -> %s (score=%.3f)", target_path, dest, score)
            target_path = dest

        self._index_chunks(target_path, index_chunks, topic)
        return {
            "path": str(target_path),
            "topic": topic or "",
            "score": f"{score:.4f}" if score is not None else "",
        }

    # ---- 新增：按 source 删除整篇论文的所有 chunks ----
    def delete_paper_by_source(self, source_path: Path) -> int:
        source_str = str(source_path.expanduser().resolve())
        # 拉取一些候选，再删 ids（简单可靠；数据量大可做 where 过滤）
        raw = self.store.get_all_ids_and_meta()
        ids_to_delete: List[str] = []
        for _id, meta in raw:
            if meta.get("source") == source_str:
                ids_to_delete.append(_id)
        if not ids_to_delete:
            return 0
        self.store.delete(ids_to_delete)
        return len(ids_to_delete)

    # ---- 新增：重建索引（清空后从 library 重新 index）----
    def rebuild_from_library(self) -> None:
        self.store.reset()
        self.index_existing(config.LIBRARY_DIR)
