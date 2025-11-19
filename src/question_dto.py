from pydantic import BaseModel

class QuestionDto(BaseModel):
    question: str
