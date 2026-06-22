import pymupdf,re, json
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
import ollama

doc = pymupdf.open("CODE_FAMILLE.pdf")
ENTETES= {"CODE DE LA FAMILLE", "Sénégalais"}

# ── SERVICE 1 EXTRATION ────────────────────────────────────────────────────────────────
pages =[]
   

# def extract_articles_from_pdf(doc):
#     articles =[]
#     current_article = None
#     current_livre   = None
#     current_chapitre = None
#     last_page_seen  = 1
#     for page_num in range(len(doc)):
#         page = doc[page_num]
#         for bloc in page.get_text("dict")["blocks"]:
#             if bloc["type"] != 0:  #ignorer les blocs images
#                 continue

#             for line in bloc["lines"]:
#                 # ── Reconstruire le texte de la ligne ─────────────────────
#                 # ── Reconstruire le texte de la ligne ─────────────────────
#                 line_text = "".join(s["text"] for s in line["spans"]).strip()
#                 if not line_text or line_text in ENTETES:
#                     continue

#                 # ── Propriétés typographiques du span dominant ─────────────
#                 dominant = next((s for s in line["spans"] if s["text"].strip()), None)
#                 if not dominant:
#                     continue
#                 is_bold = bool(dominant["flags"] & 0b10000)  # bit 4 = bold
#                 size    = dominant["size"]
#                 # ── Numéro de page : on le sait toujours (on est dans la boucle page) ──
#                 last_page_seen = page_num + 1

#                 # ── Contexte hiérarchique ──────────────────────────────────
#                 if re.match(r'^LIVRE\b', line_text) and is_bold:
#                     current_livre    = line_text
#                     current_chapitre = None
#                     continue
#                 if re.match(r'^CHAPITRE\b', line_text) and is_bold:
#                     current_chapitre = line_text
#                     continue
#                 # ── Marqueur "Article X" ───────────────────────────────────
#                 art_match = re.match(r'^Article\s+(premier|\d+)$', line_text, re.IGNORECASE)
#                 if art_match:
#                     # Sauvegarder l'article précédent
#                     if current_article:
#                         # Fusionner les tirets de coupure de colonne (ex: "désa-\nveu")
#                         current_article["contenu"] = re.sub(r'-\s+', '', current_article["contenu"]).strip()
#                         current_article["titre"]   = re.sub(r'-\s+', '', current_article["titre"]).strip()
#                         articles.append(current_article)

#                     numero = art_match.group(1)
#                     if numero.lower() == "premier":
#                         numero = "1"

#                     current_article = {
#                         "numero":   numero,
#                         "titre":    "",
#                         "contenu":  "",
#                         "page":     last_page_seen,   #  toujours connu ici
#                         "livre":    current_livre,
#                         "chapitre": current_chapitre,
#                         "_attente_titre": True         # flag interne
#                     }
#                     continue
#                 # ── Titre de l'article ─────────────────────────────────────
#                 # Le titre = span(s) bold size≈8.5 juste après "Article X"
#                 # IMPORTANT : le titre peut s'étendre sur PLUSIEURS lignes bold
#                 if current_article and current_article["_attente_titre"]:
#                     if is_bold and size < 9.0:
#                         # Accumuler les lignes bold (titre sur 2 lignes)
#                         sep = " " if current_article["titre"] else ""
#                         current_article["titre"] += sep + line_text
#                         continue
#                     else:
#                         # La ligne n'est plus bold → fin du titre, début du corps
#                         current_article["_attente_titre"] = False

#                 # ── Corps de l'article ─────────────────────────────────────
#                 if current_article:
#                     sep = " " if current_article["contenu"] else ""
#                     current_article["contenu"] += sep + line_text
#     # Ne pas oublier le dernier article
#     if current_article:
#         current_article["contenu"] = re.sub(r'-\s+', '', current_article["contenu"]).strip()
#         current_article["titre"]   = re.sub(r'-\s+', '', current_article["titre"]).strip()  #  ajout
#         articles.append(current_article)

#     # Nettoyer le flag interne avant export
#     for a in articles:
#         a.pop("_attente_titre", None)

#     return articles

# def export_json(articles, path="code_famille.json"):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(articles, f, ensure_ascii=False, indent=2)
#     print(f" {len(articles)} articles exportés → {path}")

# # ── Pipeline ──────────────────────────────────────────────────────────────────
# articles = extract_articles_from_pdf(doc)
# export_json(articles)

# print(f"\nTotal : {len(articles)} articles\n")
# for a in articles[:10]:
#     print(json.dumps(a, ensure_ascii=False, indent=2))
#     print()

# ── SERVICE 2 SEGMENTATION ────────────────────────────────────────────────────────────────

# ── Constante
# TOKENS_PAR_MOT = 1.3          # approximation : 1 mot ≈ 1.3 tokens
# MAX_TOKENS     = 500
# OVERLAP_TOKENS = 75

# def compter_tokens(text:str) -> int:
#     """Estimation rapide : nb_mots × 1.3"""
#     return int(len(text.split()) * TOKENS_PAR_MOT)

# def decouper_en_alineas(text:str)->list[str]:
#     """
#     Découpe le corps d'un article en alinéas.
#     Critères : ligne vide OU phrase se terminant par un point suivi d'une majuscule.
#     """
#     # Séparer sur double espace ou sur ". Majuscule"
#     alineas = re.split(r'\.\s+(?=[A-ZÀ-Ü])', text)
#     # Remettre le point f inal sur chaque alinéa (sauf le dernier)
#     alineas = [a.strip() + ('.' if not a.strip().endswith('.') else '')
#                for a in alineas if a.strip()]
#     return alineas

# def segmenter_article(article: dict) -> list[dict]:
#     """
#     Prend un article extrait et retourne une liste de chunks.

#     Règle :
#     - Si l'article ≤ MAX_TOKENS → 1 seul chunk (article complet)
#     - Si l'article > MAX_TOKENS → découpage en alinéas avec overlap
#     """
#     # Texte complet = titre + contenu
#     texte_complet = f"{article['titre']}. {article['contenu']}".strip()
#     nb_tokens = compter_tokens(texte_complet)

#     # Métadonnées communes à tous les chunks de cet article
#     meta = {
#         "id_article":    article["numero"],
#         "titre_article": article["titre"],
#         "page_debut":    article["page"],
#         "page_fin":      article["page"],
#         "livre":         article.get("livre"),
#         "chapitre":      article.get("chapitre"),
#     }
#     chunks = []

#     # ── Cas 1 : article court → un seul chunk
#     if nb_tokens <= MAX_TOKENS:
#         chunk_id = f"art_{article['numero']}_chunk_1"
#         chunks.append({
#             "chunk_id": chunk_id,
#             **meta,
#             "texte": texte_complet,
#             "tokens": nb_tokens
#         })
#         return chunks
    
#     # ── Cas 2 : article long → découpage en alinéas avec overlap
#     alineas = decouper_en_alineas(article["contenu"])
#     chunk_courant = article["titre"] + ". " # commencer par le titre
#     tokens_courant = compter_tokens(chunk_courant)
#     chunk_index = 1
#     buffer_overlap = [] # alinéas conservés pour le chevauchement
#     for alinea in alineas:
#         tokens_alinea = compter_tokens(alinea)

#         # Si ajouter cet alinéa dépasse la limite → sauvegarder le chunk courant
#         if tokens_courant + tokens_alinea > MAX_TOKENS and chunk_courant.strip():
#             chunk_id = f"art_{article['numero']}_chunk_{chunk_index}"
#             chunks.append({
#                 "chunk_id": chunk_id,
#                 **meta,
#                 "texte":    chunk_courant.strip(),
#                 "tokens":   compter_tokens(chunk_courant.strip()),
#             })
#             chunk_index += 1

#             # Overlap : repartir avec les derniers alinéas du chunk précédent
#             tokens_overlap = 0
#             overlap_texte  = ""
#             for a in reversed(buffer_overlap):
#                 t = compter_tokens(a)
#                 if tokens_overlap + t <= OVERLAP_TOKENS:
#                     overlap_texte  = a + " " + overlap_texte
#                     tokens_overlap += t
#                 else:
#                     break

#             chunk_courant  = overlap_texte + alinea + " "
#             tokens_courant = compter_tokens(chunk_courant)
#             buffer_overlap = [alinea]
#         else:
#             chunk_courant  += alinea + " "
#             tokens_courant += tokens_alinea
#             buffer_overlap.append(alinea)

#     # Dernier chunk restant
#     if chunk_courant.strip():
#         chunk_id = f"art_{article['numero']}_chunk_{chunk_index}"
#         chunks.append({
#             "chunk_id": chunk_id,
#             **meta,
#             "texte":    chunk_courant.strip(),
#             "tokens":   compter_tokens(chunk_courant.strip()),
#         })

#     return chunks

# def segmenter_tous_articles(articles: list[dict]) -> list[dict]:
#     """
#     Applique segmenter_article() sur tous les articles extraits.
#     Retourne la liste complète des chunks.
#     """
#     tous_chunks = []
#     for article in articles:
#         chunks = segmenter_article(article)
#         tous_chunks.extend(chunks)
#     return tous_chunks

# def export_chunks_json(chunks: list[dict], path="chunks.json"):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)
#     print(f" {len(chunks)} chunks exportés → {path}")

# # ── Pipeline complet
# with open("code_famille.json", encoding="utf-8") as f:
#     articles = json.load(f)

# chunks = segmenter_tous_articles(articles)
# export_chunks_json(chunks)

# # Aperçu
# print(f"\nTotal chunks : {len(chunks)}")
# print("\n--- Exemples ---")
# for c in chunks[43:45]:
#     print(json.dumps(c, ensure_ascii=False, indent=2))
#     print()

# # Statistiques
# courts = sum(1 for c in chunks if c["tokens"] <= MAX_TOKENS)
# longs  = len(chunks) - courts
# print(f"\nChunks simples (≤{MAX_TOKENS} tokens) : {courts}")
# print(f"Chunks issus de découpage           : {longs}")

# ── SERVICE 3 VECTORISATION

# 3.1 — Charge le modèle une seule fois en mémoire  
# paraphrase-multilingual-mpnet-base-v2 : 768 dimensions, français excellent, gratuit
# print("Chargement du modèle d'embedding...")
# modele=SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# print("Modele charge\n")

# # 3.2 — Chargement des chunks produits par le Service 2
# with open("chunks.json", encoding="utf-8") as f:
#     chunks = json.load(f)

# print(f"{len(chunks)} chunks chargés depuis chunks.json")

# # 3.3 — Nettoyage résiduel avant vectorisation
# # Supprime les numéros de page parasites glissés dans le texte (ex: "posté11 rieurement")
# def nettoyer_texte_chunk(texte: str) -> str:
#     # Supprime un nombre isolé collé à un mot (artefact PDF)
#     texte = re.sub(r'(\w)(\d{1,3})\s([a-zàâéèêëîïôùûü])', r'\1\3', texte)
#     # Supprime les numéros de page seuls sur une ligne
#     texte = re.sub(r'\b\d{1,3}\b', lambda m: '' if len(m.group()) <= 3 else m.group(), texte)
#     texte = re.sub(r'\s{2,}', ' ', texte)
#     return texte.strip()

# for chunk in chunks:
#     chunk["texte"] = nettoyer_texte_chunk(chunk["texte"])

# # 3.4 — Extraction des textes à vectoriser
# # On vectorise : titre + texte pour enrichir le sens sémantique
# textes_a_vectoriser = [
#     f"{chunk['texte']}"
#     for chunk in chunks
# ]

# # 3.5 — Calcul des embeddings (par lots pour la mémoire)
# print("\n Calcul des embeddings...")
# embeddings = modele.encode(
#     textes_a_vectoriser,
#     batch_size=32, # traiter 32 chunks à la fois
#     show_progress_bar=True,  # barre de progression
#     convert_to_numpy=True    # sortie en array numpy
# )
# print(f" {len(embeddings)} embeddings calculés | dimension : {embeddings.shape[1]}\n")

# # 3.6 — Fusion embeddings + métadonnées et export
# # On ne stocke PAS les vecteurs dans le JSON (trop lourd)
# # On les sauvegarde dans un fichier numpy séparé

# np.save("embeddings.npy", embeddings)
# print("Vecteurs sauvegardés → embeddings.npy")

# # Sauvegarder les chunks nettoyés (sans les vecteurs, ils sont dans embeddings.npy)
# with open("chunks_clean.json", "w", encoding="utf-8") as f:
#     json.dump(chunks, f, ensure_ascii=False, indent=2)
# print("✅ Chunks nettoyés sauvegardés → chunks_clean.json")

# # 3.7 — Vérification
# print("\n--- Vérification ---")
# print(f"Nombre de chunks   : {len(chunks)}")
# print(f"Nombre de vecteurs : {len(embeddings)}")
# print(f"Dimension          : {embeddings.shape[1]}")
# print(f"\nExemple chunk[0]   : {chunks[0]['chunk_id']}")
# print(f"Vecteur[0] (5 premières valeurs) : {embeddings[0][:5]}")

# # ── SERVICE 4 : INDEXATION CHROMADB
# # 4.1 — Chargement des données produites par les services précédents
# print("📦 Chargement des chunks et embeddings...")

# with open("chunks_clean.json", encoding="utf-8") as f:
#     chunks = json.load(f)

# embeddings = np.load("embeddings.npy")

# print(f"✅ {len(chunks)} chunks | {embeddings.shape} embeddings chargés\n")

# # 4.2 — Initialisation de ChromaDB (stockage local persistant)
# # PersistentClient : les données sont sauvegardées sur disque → survivent au redémarrage

# client = chromadb.PersistentClient(path="./chroma_db")

# # Supprimer la collection si elle existe déjà (utile pour relancer proprement)
# try:
#     client.delete_collection("code_famille")
#     print("  Collection existante supprimée")
# except:
#     pass

# # Créer la collection
# # embedding_function=None car on fournit nos propres vecteurs pré-calculés
# collection = client.create_collection(
#     name="code_famille",
#     metadata={"hnsw:space": "cosine"}   # distance cosinus pour la similarité sémantique
# )
# print("✅ Collection 'code_famille' créée\n")

# # 4.3 — Préparation des données pour l'insertion
# # ChromaDB attend 4 listes parallèles : ids, embeddings, documents, metadatas

# ids         = []
# vecteurs    = []
# documents   = []   # texte brut du chunk (pour la récupération)
# metadatas   = []   # payload : toutes les métadonnées filtrables

# for i, chunk in enumerate(chunks):
#     ids.append(chunk["chunk_id"])
#     vecteurs.append(embeddings[i].tolist())   # numpy array → liste Python
#     documents.append(chunk["texte"])
#     metadatas.append({
#         "id_article":    chunk["id_article"],
#         "titre_article": chunk["titre_article"],
#         "page":          chunk["page_debut"] if chunk["page_debut"] is not None else 0,
#         "livre":         chunk["livre"]   or "Non défini",
#         "chapitre":      chunk["chapitre"] or "Non défini",
#     })

# # 4.4 — Insertion par lots (batch de 100 pour la stabilité)
# BATCH_SIZE = 100
# total = len(chunks)

# print(f"Insertion de {total} chunks dans ChromaDB...")
# for debut in range(0, total, BATCH_SIZE):
#     fin = min(debut + BATCH_SIZE, total)
#     collection.add(
#         ids        = ids[debut:fin],
#         embeddings = vecteurs[debut:fin],
#         documents  = documents[debut:fin],
#         metadatas  = metadatas[debut:fin],
#     )
#     print(f"    Batch {debut//BATCH_SIZE + 1} : chunks {debut+1} → {fin} insérés")

# print(f"\n {total} chunks indexés dans ChromaDB\n")

# # 4.5 — Vérification
# count = collection.count()
# print(f"--- Vérification ---")
# print(f"Documents dans la collection : {count}")

# # Test : récupérer un chunk par son ID
# resultat = collection.get(ids=["art_1_chunk_1"], include=["documents", "metadatas"])
# print(f"\nTest get('art_1_chunk_1') :")
# print(f"  Texte    : {resultat['documents'][0][:80]}...")
# print(f"  Metadata : {resultat['metadatas'][0]}")

# # 4.6 — Test de recherche par similarité (avant-goût du Service 5)
# print("\n--- Test recherche sémantique ---")
# from sentence_transformers import SentenceTransformer

# modele = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# question_test = "Quels sont les droits de la femme mariée ?"
# vecteur_question = modele.encode(question_test).tolist()

# resultats = collection.query(
#     query_embeddings=[vecteur_question],
#     n_results=3,
#     include=["documents", "metadatas", "distances"]
# )

# print(f"Question : « {question_test} »\n")
# for j in range(len(resultats["ids"][0])):
#     print(f"  Résultat {j+1} :")
#     print(f"    chunk_id  : {resultats['ids'][0][j]}")
#     print(f"    article   : {resultats['metadatas'][0][j]['titre_article']}")
#     print(f"    page      : {resultats['metadatas'][0][j]['page']}")
#     print(f"    similarité: {1 - resultats['distances'][0][j]:.3f}")
#     print(f"    texte     : {resultats['documents'][0][j][:100]}...")
#     print()

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

# ── SERVICE 6 : RÉPONSE LLM ──────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6.1 — Construire le contexte
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6.2 — Construire le prompt
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6.3 — Appeler le LLM
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6.4 — Extraire les sources pour l'interface
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6.5 — Fonction principale du Service 6
# ─────────────────────────────────────────────────────────────────────────────

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
    print("⏳ Génération de la réponse...")
    texte_reponse = appeler_llm(messages)
    
    # 4. Préparer les sources
    sources = extraire_sources(chunks_retrouves)
    
    return {
        "question": question,
        "reponse":  texte_reponse,
        "sources":  sources,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TEST COMPLET — Pipeline Services 5 + 6
# ─────────────────────────────────────────────────────────────────────────────

# Charger ChromaDB et le modèle d'embedding (déjà initialisés avant)
# (on suppose que collection et modele_embedding sont disponibles)

question_test = "Quelles sont les conditions d'âge pour se marier au Sénégal ?"
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

