from pydantic import BaseModel

class ResponseDto(BaseModel):
    reply: str
