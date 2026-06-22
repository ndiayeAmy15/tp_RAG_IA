import re, json
# ── SERVICE 2 SEGMENTATION

# ── Constante
TOKENS_PAR_MOT = 1.3          # approximation : 1 mot ≈ 1.3 tokens
MAX_TOKENS     = 500
OVERLAP_TOKENS = 75

def compter_tokens(text:str) -> int:
    """Estimation rapide : nb_mots × 1.3"""
    return int(len(text.split()) * TOKENS_PAR_MOT)

#la fonction permet 
def decouper_en_alineas(text:str)->list[str]:
    """
    Découpe le corps d'un article en alinéas.
    Critères : ligne vide OU phrase se terminant par un point suivi d'une majuscule.
    """
    # Séparer sur double espace ou sur ". Majuscule"
    alineas = re.split(r'\.\s+(?=[A-ZÀ-Ü])', text)
    # Remettre le point final sur chaque alinéa (sauf le dernier)
    alineas = [a.strip() + ('.' if not a.strip().endswith('.') else '')
               for a in alineas if a.strip()]
    return alineas

def segmenter_article(article: dict) -> list[dict]:
    """
    Prend un article extrait et retourne une liste de chunks.

    Règle :
    - Si l'article ≤ MAX_TOKENS → 1 seul chunk (article complet)
    - Si l'article > MAX_TOKENS → découpage en alinéas avec overlap
    """
    # Texte complet = titre + contenu
    texte_complet = f"{article['titre']}. {article['contenu']}".strip()
    nb_tokens = compter_tokens(texte_complet)

    # Métadonnées communes à tous les chunks de cet article
    meta = {
        "id_article":    article["numero"],
        "titre_article": article["titre"],
        "page_debut":    article["page"],
        "page_fin":      article["page"],
        "livre":         article.get("livre"),
        "chapitre":      article.get("chapitre"),
    }
    chunks = []

    # ── Cas 1 : article court → un seul chunk
    if nb_tokens <= MAX_TOKENS:
        chunk_id = f"art_{article['numero']}_chunk_1"
        chunks.append({
            "chunk_id": chunk_id,
            **meta,
            "texte": texte_complet,
            "tokens": nb_tokens
        })
        return chunks
    
    # ── Cas 2 : article long → découpage en alinéas avec overlap
    alineas = decouper_en_alineas(article["contenu"])
    chunk_courant = article["titre"] + ". " # commencer par le titre
    tokens_courant = compter_tokens(chunk_courant)
    chunk_index = 1
    buffer_overlap = [] # alinéas conservés pour le chevauchement
    for alinea in alineas:
        tokens_alinea = compter_tokens(alinea)

        # Si ajouter cet alinéa dépasse la limite → sauvegarder le chunk courant
        if tokens_courant + tokens_alinea > MAX_TOKENS and chunk_courant.strip():
            chunk_id = f"art_{article['numero']}_chunk_{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                **meta,
                "texte":    chunk_courant.strip(),
                "tokens":   compter_tokens(chunk_courant.strip()),
            })
            chunk_index += 1

            # Overlap : repartir avec les derniers alinéas du chunk précédent
            tokens_overlap = 0
            overlap_texte  = ""
            for a in reversed(buffer_overlap):
                t = compter_tokens(a)
                if tokens_overlap + t <= OVERLAP_TOKENS:
                    overlap_texte  = a + " " + overlap_texte
                    tokens_overlap += t
                else:
                    break

            chunk_courant  = overlap_texte + alinea + " "
            tokens_courant = compter_tokens(chunk_courant)
            buffer_overlap = [alinea]
        else:
            chunk_courant  += alinea + " "
            tokens_courant += tokens_alinea
            buffer_overlap.append(alinea)

    # Dernier chunk restant
    if chunk_courant.strip():
        chunk_id = f"art_{article['numero']}_chunk_{chunk_index}"
        chunks.append({
            "chunk_id": chunk_id,
            **meta,
            "texte":    chunk_courant.strip(),
            "tokens":   compter_tokens(chunk_courant.strip()),
        })

    return chunks

def segmenter_tous_articles(articles: list[dict]) -> list[dict]:
    """
    Applique segmenter_article() sur tous les articles extraits.
    Retourne la liste complète des chunks.
    """
    tous_chunks = []
    for article in articles:
        chunks = segmenter_article(article)
        tous_chunks.extend(chunks)
    return tous_chunks

def export_chunks_json(chunks: list[dict], path="chunks.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f" {len(chunks)} chunks exportés → {path}")

# ── Pipeline complet
with open("code_famille.json", encoding="utf-8") as f:
    articles = json.load(f)

chunks = segmenter_tous_articles(articles)
export_chunks_json(chunks)

# Aperçu
print(f"\nTotal chunks : {len(chunks)}")
print("\n--- Exemples ---")
for c in chunks[43:45]:
    print(json.dumps(c, ensure_ascii=False, indent=2))
    print()

# Statistiques
courts = sum(1 for c in chunks if c["tokens"] <= MAX_TOKENS)
longs  = len(chunks) - courts
print(f"\nChunks simples (≤{MAX_TOKENS} tokens) : {courts}")
print(f"Chunks issus de découpage           : {longs}")