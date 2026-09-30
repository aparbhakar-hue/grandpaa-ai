import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(request: Request):
    data = await request.json()

    return {
        "status": "Grandpaa GridShare connection OK",
        "received": data
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
