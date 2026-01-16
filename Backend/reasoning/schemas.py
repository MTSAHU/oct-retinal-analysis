from pydantic import BaseModel
from typing import List, Optional

class SymptomInput(BaseModel):
    age: int
    gender: str
    location: str

    primary_symptom: str
    additional_symptoms: Optional[List[str]] = []

    duration: str
    severity: int  # 1–5

    family_history: bool
    chronic_condition: bool
    smoking_or_alcohol: bool