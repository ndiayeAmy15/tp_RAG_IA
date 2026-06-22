from fastapi import FastAPI
from pydantic import BaseModel

from service.service5_recherche import rechercher
from service.service6_generation import generer_reponse

from service.service3_vectorisation import modele
from service.service4_indexation import collection
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/api/chat")
def chat(req: QuestionRequest):

    chunks = rechercher(
        req.question,
        modele,
        collection,
        top_k=3
    )

    resultat = generer_reponse(
        req.question,
        chunks
    )

    return resultat