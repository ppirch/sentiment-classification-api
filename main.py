from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
from pydantic import BaseModel
from model import MySentimentModel
from enum import Enum

with open('./dumps/id2label.pkl', 'rb') as f:
    id2label = pickle.load(f)

model = MySentimentModel()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class Sentiment(str, Enum):
    q = 'q'
    neg = 'neg'
    neu = 'neu'
    pos = 'pos'

class SentimentResponse(BaseModel):
    sentiment: Sentiment

@app.get("/")
def read_root():
    return {
        "API": "Wisesight sentiment classification",
        "version": "0.1.0",
        "author": "Pakin Siwatammarat",
        "docs": "/docs"
    }

@app.post("/predict")
def predict_sentiment(text: TextInput):
    sentiment, prob = model.predict(text.text)
    return {"sentiment": sentiment, "probability": prob}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)