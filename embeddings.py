import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


LOGGER = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, text_model_path: Path, clip_model_path: Path, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading text model from %s", text_model_path)
        self.text_model = SentenceTransformer(str(text_model_path), device=self.device)

        LOGGER.info("Loading CLIP model from %s", clip_model_path)
        self.clip_model = CLIPModel.from_pretrained(str(clip_model_path), local_files_only=True)
        self.clip_processor = CLIPProcessor.from_pretrained(str(clip_model_path), local_files_only=True)
        self.clip_model.to(self.device)

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.empty((0, 1))
        vectors = self.text_model.encode(
            texts_list,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors

    def embed_clip_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.clip_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def embed_images(self, image_paths: List[Path]) -> np.ndarray:
        images: List[Image.Image] = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
        inputs = self.clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()
