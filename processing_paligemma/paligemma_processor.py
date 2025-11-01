from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    
    # Repeat <image> token 'image_seq_len' times to represent the number of image patches encoded by the vision model
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(
        image: Image,
        size: Tuple[int, int],
        resample: Image.Resampling = None,
        reducing_gap: Optional[int] = None,
) -> np.ndarray:
    # Unpack target spatial dimensions
    height, width = size

    # Resize the input image to the target (width, height) using specified resampling method
    # 'resample' controls interpolation (e.g., BICUBIC, BILINEAR, NEAREST)
    # 'reducing_gap' optimizes downscaling by minimizing aliasing artifacts
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )

    # Return resized image as a PIL Image object
    return resized_image

def rescale(
        image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    # Multiply all pixel values by the given scale factor
    # Example: scale = 1/255.0 converts pixel range from [0, 255] → [0, 1]
    rescaled_image = image * scale

    # Convert the image array to the specified data type (default: float32)
    rescaled_image = rescaled_image.astype(dtype)

    # Return the rescaled image array
    return rescaled_image   

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
) -> np.ndarray:
    # Convert mean and std to NumPy arrays (ensure same dtype as image)
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    # Normalize the image: (pixel - mean) / std
    # This ensures zero-centered pixels and scales them by per-channel variance
    image = (image - mean) / std

    # Return the normalized image array
    return image

def process_images(images: List[Image.Image],
                   size: Dict[str, int] = None,
                   resample: Image.Resampling = None,
                   rescale_factor: float = None,
                   image_mean: Optional[Union[float, List[float]]] = None,
                   image_std: Optional[Union[float, List[float]]] = None,
                   ) -> List[np.ndarray]:
    # Extract target height and width for resizing
    height, width = size[0], size[1]

    # Resize all images to a fixed spatial resolution (height x width)
    # 'resample' controls interpolation quality (e.g., bicubic for smoother results)
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # Convert each image from PIL format to NumPy array (H, W, C) with uint8 values
    images = [np.array(image) for image in images]

    # Rescale pixel values to [0, 1] range using provided rescale_factor (typically 1/255)
    images = [rescale(images, scale=rescale_factor) for image in images]

    # Normalize pixel intensities per channel using ImageNet mean and std
    # Converts raw RGB values to standardized distribution expected by pretrained models
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # Move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    # Original: [Height, Width, Channel] → Transposed: [Channel, Height, Width]
    images = [images.transpose(2, 0, 1) for image in images]

    # Return the processed list of image tensors ready for batching and model input
    return images

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        # Validate input batch size: currently supports only single (text, image) pair
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Preprocess images:
        # 1. Resize to (image_size, image_size)
        # 2. Resample using bicubic interpolation for better quality
        # 3. Normalize pixel values with ImageNet mean and std
        # 4. Rescale pixel values to [0, 1]
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor for model input
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the textual prompt
        # Example: "<bos> <image> <image> ... question text ..."
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenize the combined text (with image tokens)
        # Returns input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        # Merge image tensor and text tensors into a single batch dictionary
        return_data = {"pixel_values": pixel_values, **inputs}

        # Return final processed data used as model input
        return return_data