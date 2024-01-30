"""Modification of Multi Modal Vector Store Index from original implementation of Llama_index
- Allow input of custom Multi Modal Embedding Models

"""
import logging
from typing import Any,  Optional, Sequence, Union

from llama_index.data_structs.data_structs import IndexDict, MultiModelIndexDict
from llama_index.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.indices.utils import (
    async_embed_image_nodes,
    async_embed_nodes,
    embed_image_nodes,
    embed_nodes,
)

from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import BaseNode, ImageNode
from llama_index.service_context import ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.types import VectorStore

logger = logging.getLogger(__name__)

class MultiModalVectorStoreIndex(VectorStoreIndex):
    """Multi-Modal Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    image_namespace = "image"
    index_struct_cls = MultiModelIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[MultiModelIndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        use_async: bool = False,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        # Image-related kwargs
        image_vector_store: Optional[VectorStore] = None,
        image_embed_model: [str, MultiModalEmbedding] = "clip",
        is_image_to_text: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if isinstance(image_embed_model, str):
            image_embed_model = resolve_embed_model(image_embed_model)
        assert isinstance(image_embed_model, MultiModalEmbedding)
        self._image_embed_model = image_embed_model
        self._is_image_to_text = is_image_to_text
        storage_context = storage_context or StorageContext.from_defaults()

        if image_vector_store is not None:
            storage_context.add_vector_store(image_vector_store, self.image_namespace)

        if self.image_namespace not in storage_context.vector_stores:
            storage_context.add_vector_store(SimpleVectorStore(), self.image_namespace)

        self._image_vector_store = storage_context.vector_stores[self.image_namespace]

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            use_async=use_async,
            store_nodes_override=store_nodes_override,
            **kwargs,
        )