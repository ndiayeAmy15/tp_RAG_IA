import pymupdf,re, json
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

# ── SERVICE 5 : RECHERCHE (RETRIEVAL)
def encoder_question(question: str, modele) -> list[float]:
    """
    Transforme la question de l'utilisateur en vecteur (embedding).
    On utilise le MÊME modèle que pour les chunks (paraphrase-multilingual-mpnet-base-v2),
    sinon les vecteurs ne seraient pas comparables. modele.encode(question) → renvoie un tableau numpy de ] valeurs.
    """
    vecteur = modele.encode(question).tolist()
    return vecteur

def chercher_chunks_similaires(vecteur_question, collection, top_k=3):
    """
    Interroge ChromaDB avec le vecteur de la question.
    Retourne les top_k chunks les plus proches (similarité cosinus).
    """
    resultats = collection.query(
        query_embeddings=[vecteur_question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return resultats

def formater_resultats(resultats) -> list[dict]:
    """
    Prend la réponse brute de ChromaDB (un peu compliquée, avec des listes
    imbriquées) et la transforme en une liste simple de dictionnaires,
    plus facile à utiliser ensuite.
    """
    sorties=[]
    nb_resultats = len(resultats["ids"][0])

    for i in range(nb_resultats):
        sorties.append({
            "score":      round(1 - resultats["distances"][0][i], 3),
            "id_article": resultats["metadatas"][0][i]["id_article"],
            "titre":      resultats["metadatas"][0][i]["titre_article"],
            "texte":      resultats["documents"][0][i],
            "page":       resultats["metadatas"][0][i]["page"],
            "livre":      resultats["metadatas"][0][i]["livre"],
            "chapitre":   resultats["metadatas"][0][i]["chapitre"],
        })
    return sorties

def rechercher(question: str, modele, collection, top_k=3) -> list[dict]:
    """
    Fonction principale du Service 5.
    Combine les 3 étapes : encoder → chercher → formater.
    """
    vecteur_question = encoder_question(question,modele)
    resultats_bruts = chercher_chunks_similaires(vecteur_question,collection,top_k)
    resultats_propres = formater_resultats(resultats_bruts)
    return resultats_propres



