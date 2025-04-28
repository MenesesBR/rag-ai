from pydantic import BaseModel
from typing import List, Optional
from fastapi import UploadFile


class ClientData(BaseModel):
    client_name: str
    persona: str

class QuestionRequest(BaseModel):
    client_id: str
    question: str
