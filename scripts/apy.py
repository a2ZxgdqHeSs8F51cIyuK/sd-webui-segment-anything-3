import os
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Tuple
import gradio as gr
from PIL import Image
import numpy as np

from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from scripts.sam import sam_predict, update_mask, sam_model_list

def decode_to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, str):
        if os.path.exists(image):
            return Image.open(image)
        return decode_base64_to_image(image)
    raise ValueError("Not an image")

def encode_to_base64(image):
    if isinstance(image, str):
        return image
    if isinstance(image, Image.Image):
        return encode_pil_to_base64(image).decode()
    if isinstance(image, np.ndarray):
        return encode_pil_to_base64(Image.fromarray(image)).decode()
    raise ValueError("Invalid type")

def split_sam_gallery(gallery: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image]]:
    n = len(gallery)
    if n == 0:
        return [], [], []
    third = n // 3
    return gallery[:third], gallery[third:2*third], gallery[2*third:3*third]

def _default_sam_model() -> str:
    return sam_model_list[0] if sam_model_list else ""

def sam_api(_: gr.Blocks, app: FastAPI):
    @app.get("/sam/heartbeat")
    async def heartbeat():
        return {"msg": "Success!"}

    @app.get("/sam/sam-model")
    async def api_sam_model() -> List[str]:
        return sam_model_list

    class SamPredictRequest(BaseModel):
        sam_model_name: str = Field(default_factory=_default_sam_model)
        input_image: str
        sam_positive_points: List[List[float]] = []
        sam_negative_points: List[List[float]] = []
        text_prompt: str = ""

    @app.post("/sam/sam-predict")
    async def api_sam_predict(payload: SamPredictRequest = Body(...)) -> Any:
        print("SAM API /sam/sam-predict received request")
        payload.input_image = decode_to_pil(payload.input_image).convert("RGBA")
        sam_output_mask_gallery, sam_message = sam_predict(
            payload.sam_model_name,
            payload.input_image,
            payload.sam_positive_points,
            payload.sam_negative_points,
            payload.text_prompt
        )
        print(f"SAM API /sam/sam-predict finished: {sam_message}")
        blended, masks, masked = split_sam_gallery(sam_output_mask_gallery)
        return {
            "msg": sam_message,
            "num_masks": len(blended),
            "blended_images": list(map(encode_to_base64, blended)),
            "masks": list(map(encode_to_base64, masks)),
            "masked_images": list(map(encode_to_base64, masked)),
        }

    class DilateMaskRequest(BaseModel):
        input_image: str
        mask: str
        dilate_amount: int = 10

    @app.post("/sam/dilate-mask")
    async def api_dilate_mask(payload: DilateMaskRequest = Body(...)) -> Any:
        print("SAM API /sam/dilate-mask received request")
        payload.input_image = decode_to_pil(payload.input_image).convert("RGBA")
        payload.mask = decode_to_pil(payload.mask)
        dilate_result = list(map(encode_to_base64, update_mask(payload.mask, 0, payload.dilate_amount, payload.input_image)))
        print("SAM API /sam/dilate-mask finished")
        return {"blended_image": dilate_result[0], "mask": dilate_result[1], "masked_image": dilate_result[2]}

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(sam_api)
except Exception as e:
    print(f"SAM Web UI API failed: {e}")
