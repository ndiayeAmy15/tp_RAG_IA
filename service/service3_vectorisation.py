import pymupdf,re, json
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb

# ── SERVICE 3 VECTORISATION

# 3.1 — Charge le modèle une seule fois en mémoire  
# paraphrase-multilingual-mpnet-base-v2 : 768 dimensions, français excellent, gratuit
print("Chargement du modèle d'embedding...")
modele=SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Modele charge\n")

# 3.2 — Chargement des chunks produits par le Service 2
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"{len(chunks)} chunks chargés depuis chunks.json")

# 3.3 — Nettoyage résiduel avant vectorisation
# Supprime les numéros de page parasites glissés dans le texte (ex: "posté11 rieurement")
def nettoyer_texte_chunk(texte: str) -> str:
    # Supprime un nombre isolé collé à un mot (artefact PDF)
    texte = re.sub(r'(\w)(\d{1,3})\s([a-zàâéèêëîïôùûü])', r'\1\3', texte)
    # Supprime les numéros de page seuls sur une ligne
    texte = re.sub(r'\b\d{1,3}\b', lambda m: '' if len(m.group()) <= 3 else m.group(), texte)
    texte = re.sub(r'\s{2,}', ' ', texte)
    return texte.strip()

for chunk in chunks:
    chunk["texte"] = nettoyer_texte_chunk(chunk["texte"])

# 3.4 — Extraction des textes à vectoriser
# On vectorise : titre + texte pour enrichir le sens sémantique
textes_a_vectoriser = [
    f"{chunk['texte']}"
    for chunk in chunks
]

# 3.5 — Calcul des embeddings (par lots pour la mémoire)
print("\n Calcul des embeddings...")
embeddings = modele.encode(
    textes_a_vectoriser,
    batch_size=32, # traiter 32 chunks à la fois
    show_progress_bar=True,  # barre de progression
    convert_to_numpy=True    # sortie en array numpy
)
print(f" {len(embeddings)} embeddings calculés | dimension : {embeddings.shape[1]}\n")

# 3.6 — Fusion embeddings + métadonnées et export
# On ne stocke PAS les vecteurs dans le JSON (trop lourd)
# On les sauvegarde dans un fichier numpy séparé

np.save("embeddings.npy", embeddings)
print("Vecteurs sauvegardés → embeddings.npy")

# Sauvegarder les chunks nettoyés (sans les vecteurs, ils sont dans embeddings.npy)
with open("chunks_clean.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print("✅ Chunks nettoyés sauvegardés → chunks_clean.json")

# 3.7 — Vérification
print("\n--- Vérification ---")
print(f"Nombre de chunks   : {len(chunks)}")
print(f"Nombre de vecteurs : {len(embeddings)}")
print(f"Dimension          : {embeddings.shape[1]}")
print(f"\nExemple chunk[0]   : {chunks[0]['chunk_id']}")
print(f"Vecteur[0] (5 premières valeurs) : {embeddings[0][:5]}")