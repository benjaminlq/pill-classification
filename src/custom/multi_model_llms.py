import base64
import io

from PIL import Image
from typing import Optional, Literal, Tuple, Union, Literal, Sequence, Dict, Any

import httpx
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.llms.openai_utils import (
    from_openai_message,
    resolve_openai_credentials,
    to_openai_message_dicts,
)
from llama_index.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.multi_modal_llms.openai_utils import (
    GPT4V_MODELS
)
from llama_index.schema import ImageDocument

def encode_image(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    with open(image_path, "rb") as image_file:
        img_64_str = base64.b64encode(image_file.read()).decode('utf-8')
    
    if resize:
        img_data = base64.b64decode(img_64_str)
        img = Image.open(io.BytesIO(img_data))
        h, w = img.size
        if isinstance(resize, int):
            resize = (resize, resize)
        elif resize == "auto":
            if h < 512 and w < 512:
                resize = (h, w)
            elif h > w:
                resize = (512, int(w / (h/512)))
            else:
                resize = (int(h / (w/512)), 512)
                
        resized_img = img.resize(resize)
        
        # Save the resized image to a buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode the resized image to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    else:
        return img_64_str
    
def generate_img_url(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    encoded_image = encode_image(image_path, resize=resize)
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    return image_url

def generate_openai_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
    image_detail: Optional[str] = "high",
    resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None
) -> ChatMessage:
    # if image_documents is empty, return text only chat message
    if image_documents is None:
        return ChatMessage(role=role, content=prompt)

    # if image_documents is not empty, return text with images chat message
    completion_content = [{"type": "text", "text": prompt}]
    for image_document in image_documents:
        image_content: Dict[str, Any] = {}
        mimetype = image_document.image_mimetype or "image/jpeg"
        if image_document.image and image_document.image != "":
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mimetype};base64,{image_document.image}",
                    "detail": image_detail,
                },
            }
        elif image_document.image_path and image_document.image_path != "":
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mimetype};base64,{base64_image}",
                    "detail": image_detail,
                },
            }
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            base64_image = encode_image(image_document.metadata["file_path"])
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": image_detail,
                },
            }

        completion_content.append(image_content)
    return ChatMessage(role=role, content=completion_content)




class OpenAIMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from OpenAI.")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: Optional[int] = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt",
        gt=0,
    )
    context_window: Optional[int] = Field(
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    image_detail: str = Field(
        description="The level of details for image in API calls. Can be low, high, or auto"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries.",
        gte=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        gte=0,
    )
    api_key: str = Field(default=None, description="The OpenAI API key.", exclude=True)
    api_base: str = Field(default=None, description="The base URL for OpenAI API.")
    api_version: str = Field(description="The API version for OpenAI API.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    default_headers: Dict[str, str] = Field(
        default=None, description="The default headers for API requests."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _client: SyncOpenAI = PrivateAttr()
    _aclient: AsyncOpenAI = PrivateAttr()
    _http_client: Optional[httpx.Client] = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: Optional[int] = 300,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: Optional[int] = DEFAULT_CONTEXT_WINDOW,
        max_retries: int = 3,
        timeout: float = 60.0,
        image_detail: str = "low",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)
        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_kwargs=additional_kwargs or {},
            context_window=context_window,
            image_detail=image_detail,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            callback_manager=callback_manager,
            default_headers=default_headers,
            **kwargs,
        )
        self._http_client = http_client
        self._client, self._aclient = self._get_clients(**kwargs)

    def _get_clients(self, **kwargs: Any) -> Tuple[SyncOpenAI, AsyncOpenAI]:
        client = SyncOpenAI(**self._get_credential_kwargs())
        aclient = AsyncOpenAI(**self._get_credential_kwargs())
        return client, aclient

    @classmethod
    def class_name(cls) -> str:
        return "openai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            num_output=self.max_new_tokens or DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "http_client": self._http_client,
            "timeout": self.timeout,
            **kwargs,
        }

    ## Change this method
    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageDocument],
        **kwargs: Any,
    ) -> List[ChatCompletionMessageParam]:
        return to_openai_message_dicts(
            [
                generate_openai_multi_modal_chat_message(
                    prompt=prompt,
                    role=role,
                    image_documents=image_documents,
                    image_detail=self.image_detail,
                )
            ]
        )

    # Model Params for OpenAI GPT4V model.
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in GPT4V_MODELS:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: {list(GPT4V_MODELS.keys())}"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_new_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = self.max_new_tokens
        return {**base_kwargs, **self.additional_kwargs}

    def _complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dict = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER, image_documents=image_documents
        )
        response = self._client.chat.completions.create(
            messages=message_dict,
            stream=False,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.choices[0].message.content,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        all_kwargs = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(messages)
        response = self._client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **all_kwargs,
        )
        openai_message = response.choices[0].message
        message = from_openai_message(openai_message)

        return ChatResponse(
            message=message,
            raw=response,
            additional_kwargs=self._get_response_token_counts(response),
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt, image_documents, **kwargs)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)
