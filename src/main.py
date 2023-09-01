"""FastAPI entry file."""
import os

import torch
import uvicorn
from fastapi import FastAPI, Response
from starlette.middleware.cors import CORSMiddleware

from gpt2 import GPT_MODEL_PATH, GPTClassifier, train_gpt
from utils import text_preprocessing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_model = GPTClassifier(device, load_from_path=True)


@app.get("/api/v1/train")
async def model_train():
    """Train model if not found."""
    if os.path.exists(GPT_MODEL_PATH):
        return Response(
            content=f"Model trained: {GPT_MODEL_PATH}", media_type="text"
        )
    else:
        return train_gpt(device, gpt_model)


@app.post("/api/v1/predict")
async def model_predict(text: str):
    """Use model for prediction."""
    return Response(
        content=gpt_model.prediction(text_preprocessing(text)),
        media_type="text",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
