# Repository Guidelines

## Project Structure & Modules
- Core CLI entry: `main.py` with subcommands (`add_paper`, `organize`, `search_paper`, `search_image`).
- Feature modules: `paper_manager.py`, `image_manager.py`, `embeddings.py`, `vector_store.py`, `pdf_utils.py`, `config.py`.
- Data/models: `datasets/papers`, `datasets/images`, `all-MiniLM-L6-v2`, `models--openai--clip-vit-base-patch32`.
- Outputs: classified papers in `library/`; vector stores in `storage/chroma_papers` and `storage/chroma_images`.

## Build, Run, and Development
- Install dependencies: `pip install -r requirements.txt`.
- CLI help: `python main.py --help`.
- Common commands:
  - `python main.py add_paper <pdf> --topics "CV,NLP,RL"`: copy/classify a single PDF into `library/<topic>/` and index it.
  - `python main.py organize datasets/papers --topics "CV,NLP,RL"`: batch classify + index PDFs.
  - `python main.py search_paper "<query>" --top_k 5`: semantic paper search (auto-indexes library if empty).
  - `python main.py search_image "<query>" --top_k 3`: text-to-image search (auto-indexes `datasets/images` if empty).

## Coding Style & Naming
- Python 3.8+; use type hints for public functions.
- Keep code ASCII; logging via `logging` module, default INFO level in CLI.
- Functions and variables use `snake_case`; classes use `CamelCase`.
- Avoid inline business logic in `main.py`; put behavior in modules and keep CLI handlers thin.

## Testing Guidelines
- No formal test suite is present; prefer adding lightweight CLI-driven sanity checks or unit tests under a future `tests/` directory.
- If adding tests, name files `test_*.py` and ensure they do not depend on network; use local models/data only.

## Commit & Pull Request Guidelines
- Commit messages: short imperative summary (e.g., `add image search indexing`, `fix pdf chunking`).
- PRs should describe the feature/fix, list key commands run, and mention any model/data expectations (e.g., needs `all-MiniLM-L6-v2` present).
- Include screenshots or sample outputs for UX-affecting changes (e.g., search results) when relevant.
