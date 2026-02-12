from typing import Optional
from sqlmodel import SQLModel, Field, Relationship

class ImageBase(SQLModel):
    filename: str

class Image(ImageBase, table=True):
    id: str = Field(default=None, primary_key=True)
    coins: list["Coin"] = Relationship(back_populates="image")

class CoinBase(SQLModel):
    center_x: float
    center_y: float
    radius: float
    is_slanted: bool
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

class Coin(CoinBase, table=True):
    id: str = Field(default=None, primary_key=True)
    image_id: str = Field(foreign_key="image.id")
    image: Optional[Image] = Relationship(back_populates="coins")

class CoinRead(CoinBase):
    id: str

class ImageRead(ImageBase):
    id: str
    coins: list[CoinRead] = []

