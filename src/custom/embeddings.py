import logging

from typing import Any, Optional, List
from llama_index.schema import ImageType
from llama_index.embeddings import ClipEmbedding
from llama_index.embeddings.base import Embedding

AVAILABLE_CLIP_MODELS = (
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
)
DEFAULT_CLIP_MODEL = "ViT-B/32"

logger = logging.getLogger(__name__)

class CustomClipEmbedding(ClipEmbedding):
    def __init__(
        self,
        *,
        embed_batch_size: int = 32,
        model_name: str = DEFAULT_CLIP_MODEL,
        download_root: Optional[str] = None,
        **kwargs: Any,
    ):
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size}  must be > 0.")

        try:
            import clip
            import torch
        except ImportError:
            raise ImportError(
                "ClipEmbedding requires `pip install git+https://github.com/openai/CLIP.git` and torch."
            )

        super().__init__(
            embed_batch_size=embed_batch_size, model_name=model_name, **kwargs
        )

        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model_name not in AVAILABLE_CLIP_MODELS:
                raise ValueError(
                    f"Model name {self.model_name} is not available in CLIP."
                )
            self._model, self._preprocess = clip.load(
                self.model_name, device=self._device, download_root=download_root
            )

        except Exception as e:
            print(f"Error while loading clip model.")
            raise ValueError("Unable to fetch the requested embeddings model") from e

    def _get_image_embeddings(self, img_file_paths: List[ImageType]) -> List[Embedding]:
        """
        Embed the input sequence of image synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_image_embedding
        # return [self._get_image_embedding(img_file_path) for img_file_path in img_file_paths]

        # Re-implement for batch processing on CLIP model
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "ClipEmbedding requires `pip install torch` and `pip install pillow`."
            )

        img_list = []
        with torch.no_grad():
            for file_path in img_file_paths:
                image = self._preprocess(Image.open(file_path))
                img_list.append(image)
                img_batch = torch.stack(img_list, dim=0).to(self._device)
                image_features = self._model.encode_image(img_batch)

        return image_features.tolist()
    
