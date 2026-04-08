from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    dirt: List[float]
    sun: float
    battery: float
    hour: int

class Action(BaseModel):
    action: str

class Reward(BaseModel):
    score: float
