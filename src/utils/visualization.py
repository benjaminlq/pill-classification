from typing import List, Union
import matplotlib.pyplot as plt
from llama_index.schema import ImageDocument
from PIL import Image

def plot_images(images: List[Union[str, ImageDocument]]):
    image_paths = []
    for image in images:
        if isinstance(image, ImageDocument):
            image_paths.append(image.metadata["file_path"])
        elif isinstance(image, str):
            image_paths.append(image)
        else:
            ValueError("Invalid type of image")

    images_shown = 0
    plt.figure(figsize=(8, 8 * len(image_paths)))
    for img_path in image_paths:
        image = Image.open(img_path)

        plt.subplot(1, len(image_paths), images_shown + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        images_shown += 1