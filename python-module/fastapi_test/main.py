from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

embeddings = np.genfromtxt("embeddings.txt", delimiter=',')
embeddings = embeddings[1:10001]


class RequestModel(BaseModel):
    embedding: List[float]


@app.post("/")
def endpoint(request_model: RequestModel):
    embeddings1 = np.expand_dims(np.array(request_model.embedding), axis=0)

    return {"result": str(test(embeddings1))}


def test(embeddings1):
    embeddings2 = embeddings

    embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    # print(np.argmax(dist));
    return np.max(dist)