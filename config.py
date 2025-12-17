from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
PAPER_DIR = DATA_DIR / "papers"
IMAGE_DIR = DATA_DIR / "images"

LIBRARY_DIR = BASE_DIR / "library"

STORAGE_DIR = BASE_DIR / "storage"
PAPER_DB = STORAGE_DIR / "chroma_papers"
IMAGE_DB = STORAGE_DIR / "chroma_images"

# 本地模型地址
TEXT_MODEL_PATH = BASE_DIR / "all-MiniLM-L6-v2"
CLIP_MODEL_PATH = BASE_DIR / "models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# Defaults
DEFAULT_TOP_K = 10
PDF_CHUNK_SIZE = 300

# ---- 分类&抽取增强 ----
PDF_CLASSIFY_MAX_PAGES = 6          # 分类只看前5页（索引仍可全篇）
PDF_STOP_AT_REFERENCES = True       # 遇到 references/bibliography 截断（用于分类）
CLASSIFY_TOP_M_CHUNKS = 6           # 每个 topic 取最相似的 top-m chunks
TOPIC_DESC = {
    "CV": "Computer Vision, images, detection, segmentation, recognition, 3D, pose, camera, ViT, CNN",
    "NLP": "Natural Language Processing, text, transformer, token, language model, embedding, retrieval, seq2seq",
    "RL": "Reinforcement Learning, agent, environment, reward, policy, value function, MDP, PPO, Q-learning",
    "RLHF": "Reinforcement Learning from Human Feedback, preference model, reward model, alignment, DPO, PPO",
    "Vision": "Vision models, visual representation, image understanding, multimodal, CLIP, ViT",
}

# ---- Search filtering ----
FILTER_REFERENCE_CHUNKS = True
SEARCH_FETCH_MULTIPLIER = 8  # query 时先取 top_k*6，再过滤 refs
# ---- Search diversification (group by paper) ----
SNIPPETS_PER_PAPER = 2          # 每篇论文展示几个片段
