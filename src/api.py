from fastapi import FastAPI

from agent import agent # pyright: ignore[reportUnknownVariableType]
from question_dto import QuestionDto
from response_dto import ResponseDto

app = FastAPI()

@app.post('/')
def ask_question(payload: QuestionDto) -> ResponseDto:
    result = agent.invoke( # pyright: ignore[reportUnknownMemberType]
        {"messages": [{"role": "user", "content": payload.question}]},
        stream_mode="values",
    )

    return result["structured_response"]

