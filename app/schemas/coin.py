from pydantic import BaseModel
from typing import List, Tuple

class CoinBase(BaseModel):
    """
    Base schema for a coin, containing common attributes.
    """
    coin_id: str
    center_x: float
    center_y: float
    radius: float
    is_slanted: bool


class CoinResponse(CoinBase):
    """
    Schema for the response when a coin is detected, including additional attributes.
    """
    bbox: Tuple[float, float, float, float]  # (x, y, width, height)
    mask_path: str = None # Future safe: link to mask file if we decide to save it

class ImageResponse(BaseModel):
    """
    Schema for the response when an image is processed, containing a list of detected coins.
    """
    image_id: str
    filename: str
    coin_count: int
    coins: List[CoinResponse]

class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    detail: str