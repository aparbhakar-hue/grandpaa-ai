import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


class RequestData(BaseModel):
    input: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/")
def run(data: RequestData):
    global model

    if model is None:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    return {
        "status": "Grandpaa Voice Worker Ready",
        "device": device
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
