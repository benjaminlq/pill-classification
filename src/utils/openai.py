from typing import Optional, Sequence, Union, Tuple, Literal, Dict, Any
from llama_index.schema import ImageDocument
from llama_index.core.llms.types import ChatMessage
from PIL import Image
import io
import base64
import math

def encode_image(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    if resize:
        img = Image.open(image_path)
        h, w = img.size
        if isinstance(resize, int):
            resize = resize_with_same_aspect_ratio(h, w, resize)
        elif resize == "auto":
            resize = resize_with_same_aspect_ratio(h, w, 512)

        resized_img = img.resize(resize)

        # Save the resized image to a buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode the resized image to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    else:
        with open(image_path, "rb") as image_file:
            img_64_str = base64.b64encode(image_file.read()).decode('utf-8')
        return img_64_str

def generate_img_url(image_path: str, resize: Optional[Union[int, Tuple[int, int], Literal["auto"]]] = None):
    encoded_image = encode_image(image_path, resize=resize)
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    return image_url

def resize_with_same_aspect_ratio(height: int, width: int, max_dimension: int = 1024) -> Tuple[int, int]:
    if height < max_dimension and width < max_dimension:
        resize = (height, width)
    elif height > width:
        resize = (max_dimension, int(width / (height/max_dimension)))
    else:
        resize = (int(height / (width/max_dimension)), max_dimension)
    return resize

def calculate_token_img(
    w: int, h: int, quality: Literal["low", "high"] = "high"
    ) -> int:
    if quality == "low":
        return 85
    
    if w > 2048 and h > 2048:
        if w <= h:
            w = int(w / (h / 2048))
            h = 2048
        else:
            h = int(h / (w/2048))
            w = 2048
    elif w > 2048:
        h = int(h / (w/2048))
        w = 2048
    elif h > 2048:
        w = int(w / (h / 2048))
        h = 2048
        
    if w > 512 and h > 512:
        if w >= h:
            w = int(w / (h/768))
            h = 768
        else:
            h = int(h / (w/768))
            w = 768
        
    no_of_tiles = math.ceil(w/512) * math.ceil(h/512)
    return 170 * no_of_tiles + 85

def generate_openai_vision_llamaindex_chat_message(
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
        if image_document.image_path and image_document.image_path != "":
            base64_image = encode_image(image_document.image_path, resize=resize)
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
            base64_image = encode_image(image_document.metadata["file_path"], resize=resize)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": image_detail,
                },
            }

        completion_content.append(image_content)
    return ChatMessage(role=role, content=completion_content)