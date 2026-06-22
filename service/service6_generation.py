import pymupdf,re, json
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from service.service5_recherche import rechercher
import ollama
from service.service3_vectorisation import modele
from service.service4_indexation import collection

# ── SERVICE 6 : RÉPONSE LLM
# ÉTAPE 6.1 — Construire le contexte
# Les chunks retrouvés par le Service 5 ressemblent à :
# [
#   {"score": 0.87, "id_article": "7", "titre": "Femme mariée", 
#    "texte": "...", "page": 7, "livre": "LIVRE PREMIER", "chapitre": "..."},
#   ...
# ]
# On les transforme en un texte lisible pour le LLM

def construire_contexte(chunks_retrouves: list[dict]) -> str:
    """
    Transforme la liste de chunks en un bloc de texte structuré
    que le LLM va pouvoir lire et citer.
    
    Exemple de sortie :
    === Article 7 – Femme mariée (Page 7) ===
    La femme mariée conserve son nom...
    
    === Article 214 – Effets du mariage (Page 47) ===
    Le mari est le chef de famille...
    """
    lignes = []
    for chunk in chunks_retrouves:
        ligne = (
            f"=== Article {chunk['id_article']} – "
            f"{chunk['titre']} (Page {chunk['page']}) ===\n"
            f"{chunk['texte']}"
        )
        lignes.append(ligne)
    
    # Joindre tous les articles avec une ligne vide entre chaque
    return "\n\n".join(lignes)

# ÉTAPE 6.2 — Construire le prompt
# Le prompt = les instructions qu'on envoie au LLM
# Il y a deux parties :
#   - system : le "rôle" du LLM, ses règles de comportement
#   - user   : la question + les articles

def construire_prompt(question: str, contexte: str) -> list[dict]:
    """
    Retourne une liste de messages au format attendu par Ollama/OpenAI.
    C'est comme une conversation avec deux interlocuteurs :
      - "system" : les règles imposées au LLM
      - "user"   : ce que l'utilisateur envoie
    """
    
    system_prompt = """Vous êtes un expert juridique spécialisé dans le Code de la Famille sénégalais.

RÈGLES STRICTES :
1. Répondez UNIQUEMENT en vous basant sur les articles fournis ci-dessous.
2. Citez systématiquement chaque article utilisé : numéro, titre, et page.
3. Si la réponse ne figure PAS dans les articles fournis, dites explicitement :
   "Je ne trouve pas cette information dans les articles disponibles."
4. Ne faites pas référence à d'autres lois ou connaissances externes.
5. Structurez votre réponse : explication claire, puis sources utilisées."""

    user_prompt = f"""ARTICLES DU CODE DE LA FAMILLE :
{contexte}

QUESTION : {question}

Répondez en structurant ainsi :
1. Réponse claire et pédagogique
2. Section "SOURCES UTILISÉES" avec pour chaque article :
   - Numéro et titre
   - Page dans le PDF
   - Citation courte du passage utilisé"""

    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_prompt},
    ]

# ÉTAPE 6.3 — Appeler le LLM
def appeler_llm(messages: list[dict], modele_llm: str = "mistral") -> str:
    """
    Envoie le prompt à Ollama (qui tourne localement sur ton PC).
    Retourne la réponse du LLM sous forme de texte.
    """
    reponse = ollama.chat(
        model=modele_llm,  
        messages=messages
    )
    # La réponse est dans reponse["message"]["content"]
    return reponse["message"]["content"]

def extraire_sources(chunks_retrouves: list[dict]) -> list[dict]:
    """
    Prépare la liste des sources à afficher dans l'interface.
    Chaque source contient les infos nécessaires pour le lien cliquable.
    """
    sources = []
    for chunk in chunks_retrouves:
        sources.append({
            "numero":  chunk["id_article"],
            "titre":   chunk["titre"],
            "page":    chunk["page"],
            "score":   chunk["score"],
            "livre":   chunk["livre"],
            "chapitre": chunk["chapitre"],
            # Lien direct vers la page dans le PDF (pour PDF.js)
            "lien_pdf": f"CODE_FAMILLE.pdf#page={chunk['page']}"
        })
    return sources

# ÉTAPE 6.5 — Fonction principale du Service 6


def generer_reponse(question: str, chunks_retrouves: list[dict]) -> dict:
    """
    Fonction principale : prend une question + les chunks du Service 5,
    retourne la réponse complète avec sources.
    """
    # 1. Construire le contexte à partir des chunks
    contexte = construire_contexte(chunks_retrouves)
    
    # 2. Construire le prompt
    messages = construire_prompt(question, contexte)
    
    # 3. Appeler le LLM
    print("Génération de la réponse...")
    texte_reponse = appeler_llm(messages)
    
    # 4. Préparer les sources
    sources = extraire_sources(chunks_retrouves)
    
    return {
        "question": question,
        "reponse":  texte_reponse,
        "sources":  sources,
    }

# TEST COMPLET — Pipeline Services 5 + 6
# Charger ChromaDB et le modèle d'embedding (déjà initialisés avant)
# (on suppose que collection et modele_embedding sont disponibles)

question_test = "Quel nom porte l'enfant légitime ?"
print(f"Question : {question_test}\n")

# Service 5 : recherche
chunks_retrouves = rechercher(question_test, modele, collection, top_k=3)

print(f"📄 {len(chunks_retrouves)} articles retrouvés :")
for c in chunks_retrouves:
    print(f"   Art.{c['id_article']} – {c['titre']} | score={c['score']} | page={c['page']}")

# Service 6 : génération
resultat = generer_reponse(question_test, chunks_retrouves)

print("\n" + "="*60)
print("RÉPONSE DU CHATBOT :")
print("="*60)
print(resultat["reponse"])

print("\n" + "="*60)
print("SOURCES UTILISÉES :")
print("="*60)
for s in resultat["sources"]:
    print(f"  📌 Art.{s['numero']} – {s['titre']}")
    print(f"     Page : {s['page']} | Score : {s['score']}")
    print(f"     Lien : {s['lien_pdf']}")
    print()