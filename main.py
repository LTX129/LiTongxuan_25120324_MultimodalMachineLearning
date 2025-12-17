import argparse
import logging
from pathlib import Path
from typing import Tuple

import config
from embeddings import EmbeddingManager
from image_manager import ImageManager
from paper_manager import PaperManager
from vector_store import VectorStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def build_managers() -> Tuple[PaperManager, ImageManager]:
    embedding_manager = EmbeddingManager(config.TEXT_MODEL_PATH, config.CLIP_MODEL_PATH)
    paper_store = VectorStore(config.PAPER_DB, "papers")
    image_store = VectorStore(config.IMAGE_DB, "images")
    paper_manager = PaperManager(embedding_manager, paper_store)
    image_manager = ImageManager(embedding_manager, image_store)
    return paper_manager, image_manager


def handle_add_paper(args: argparse.Namespace, paper_manager: PaperManager) -> None:
    result = paper_manager.add_paper(Path(args.path), args.topics)
    LOGGER.info("Added %s", result)


def handle_organize(args: argparse.Namespace, paper_manager: PaperManager) -> None:
    results = paper_manager.organize_folder(Path(args.folder), args.topics)
    LOGGER.info("Organized %d papers", len(results))


def handle_search_paper(args: argparse.Namespace, paper_manager: PaperManager) -> None:
    if paper_manager.store.count() == 0:
        LOGGER.info("Paper index empty, indexing existing PDFs from library %s", config.LIBRARY_DIR)
        paper_manager.index_existing(config.LIBRARY_DIR)
    results = paper_manager.search_grouped(args.query, args.top_k)
    for rank, item in enumerate(results, start=1):
        print(f"[{rank}] {item['source']} (topic={item['topic']}, score={item['best_score']:.3f})")
        for s_idx, snip in enumerate(item["snippets"], start=1):
            text = (snip["text"] or "").replace("\n", " ")
            print(f"  ({s_idx}) [chunk {snip.get('chunk_idx', '')}] score={snip['score']:.3f}: {text[:220]}...")
        print("-" * 60)


def handle_search_image(args: argparse.Namespace, image_manager: ImageManager) -> None:
    if image_manager.store.count() == 0:
        LOGGER.info("Image index empty, indexing %s", config.IMAGE_DIR)
        image_manager.index_folder(config.IMAGE_DIR)
    results = image_manager.search_by_text(args.query, args.top_k)
    for rank, item in enumerate(results, start=1):
        print(f"[{rank}] {item['path']} ({item['caption']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add_paper", help="Add and optionally classify a PDF")
    add_parser.add_argument("path", help="Path to PDF file")
    add_parser.add_argument("--topics", help="Comma-separated topics for classification", default=None)

    organize_parser = subparsers.add_parser("organize", help="Batch organize PDFs in a folder")
    organize_parser.add_argument("folder", help="Folder containing PDFs")
    organize_parser.add_argument("--topics", required=True, help='Topics, e.g. "CV,NLP,RL"')

    search_paper_parser = subparsers.add_parser("search_paper", help="Semantic search over papers")
    search_paper_parser.add_argument("query", help="Natural language query")
    search_paper_parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)

    search_image_parser = subparsers.add_parser("search_image", help="Search images with text")
    search_image_parser.add_argument("query", help="Natural language query for the target image")
    search_image_parser.add_argument("--top_k", type=int, default=3)
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")

    rebuild_parser = subparsers.add_parser("rebuild_index", help="Clear and rebuild paper/image index")

    remove_parser = subparsers.add_parser("remove_paper", help="Remove a paper from index by its path")
    remove_parser.add_argument("path", help="Path to a PDF (must match stored 'source' path)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paper_manager, image_manager = build_managers()

    if args.command == "add_paper":
        handle_add_paper(args, paper_manager)
    elif args.command == "organize":
        handle_organize(args, paper_manager)
    elif args.command == "search_paper":
        handle_search_paper(args, paper_manager)
    elif args.command == "search_image":
        handle_search_image(args, image_manager)
    elif args.command == "stats":
        print(f"Papers indexed chunks: {paper_manager.store.count()}")
        print(f"Images indexed: {image_manager.store.count()}")
        print(f"Library dir: {config.LIBRARY_DIR}")
        print(f"Paper DB: {config.PAPER_DB}")
        print(f"Image DB: {config.IMAGE_DB}")

    elif args.command == "rebuild_index":
        LOGGER.info("Rebuilding paper index from library %s", config.LIBRARY_DIR)
        paper_manager.rebuild_from_library()
        LOGGER.info("Rebuilding image index from %s", config.IMAGE_DIR)
        image_manager.store.reset()
        image_manager.index_folder(config.IMAGE_DIR)
        LOGGER.info("Done.")

    elif args.command == "remove_paper":
        removed = paper_manager.delete_paper_by_source(Path(args.path))
        LOGGER.info("Removed %d chunks for %s", removed, args.path)


if __name__ == "__main__":
    main()

