import pymupdf,re, json
import numpy as np
import chromadb

# ── SERVICE 4 : INDEXATION CHROMADB
# 4.1 — Chargement des données produites par les services précédents
print(" Chargement des chunks et embeddings...")

with open("chunks_clean.json", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("embeddings.npy")

print(f"✅ {len(chunks)} chunks | {embeddings.shape} embeddings chargés\n")

# 4.2 — Initialisation de ChromaDB (stockage local persistant)
# PersistentClient : les données sont sauvegardées sur disque → survivent au redémarrage

client = chromadb.PersistentClient(path="./chroma_db")

# Supprimer la collection si elle existe déjà (utile pour relancer proprement)
try:
    client.delete_collection("code_famille")
    print("  Collection existante supprimée")
except:
    pass

# Créer la collection
# embedding_function=None car on fournit nos propres vecteurs pré-calculés
collection = client.create_collection(
    name="code_famille",
    metadata={"hnsw:space": "cosine"}   # distance cosinus pour la similarité sémantique
)
print("✅ Collection 'code_famille' créée\n")

# 4.3 — Préparation des données pour l'insertion
# ChromaDB attend 4 listes parallèles : ids, embeddings, documents, metadatas

ids         = []
vecteurs    = []
documents   = []   # texte brut du chunk (pour la récupération)
metadatas   = []   # payload : toutes les métadonnées filtrables

for i, chunk in enumerate(chunks):
    ids.append(chunk["chunk_id"])
    vecteurs.append(embeddings[i].tolist())   # numpy array → liste Python
    documents.append(chunk["texte"])
    metadatas.append({
        "id_article":    chunk["id_article"],
        "titre_article": chunk["titre_article"],
        "page":          chunk["page_debut"] if chunk["page_debut"] is not None else 0,
        "livre":         chunk["livre"]   or "Non défini",
        "chapitre":      chunk["chapitre"] or "Non défini",
    })
# 4.4 — Insertion par lots (batch de 100 pour la stabilité)
BATCH_SIZE = 100
total = len(chunks)
print(f"Insertion de {total} chunks dans ChromaDB...")
for debut in range(0, total, BATCH_SIZE):
    fin = min(debut + BATCH_SIZE, total)
    collection.add(
        ids        = ids[debut:fin],
        embeddings = vecteurs[debut:fin],
        documents  = documents[debut:fin],
        metadatas  = metadatas[debut:fin],
    )
    print(f"    Batch {debut//BATCH_SIZE + 1} : chunks {debut+1} → {fin} insérés")

print(f"\n {total} chunks indexés dans ChromaDB\n")

# 4.5 — Vérification
count = collection.count()
print(f"--- Vérification ---")
print(f"Documents dans la collection : {count}")

# Test : récupérer un chunk par son ID
resultat = collection.get(ids=["art_1_chunk_1"], include=["documents", "metadatas"])
print(f"\nTest get('art_1_chunk_1') :")
print(f"  Texte    : {resultat['documents'][0][:80]}...")
print(f"  Metadata : {resultat['metadatas'][0]}")
