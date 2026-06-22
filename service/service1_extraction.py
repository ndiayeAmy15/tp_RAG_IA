import pymupdf,re, json

doc = pymupdf.open("CODE_FAMILLE.pdf")
ENTETES= {"CODE DE LA FAMILLE", "Sénégalais"}

# ── SERVICE 1 EXTRATION
pages =[]

def extract_articles_from_pdf(doc):
    articles =[]
    current_article = None
    current_livre   = None
    current_chapitre = None
    last_page_seen  = 1
    for page_num in range(len(doc)):
        page = doc[page_num]
        for bloc in page.get_text("dict")["blocks"]:
            if bloc["type"] != 0:  #ignorer les blocs images
                continue

            for line in bloc["lines"]:
                # ── Reconstruire le texte de la ligne
                line_text = "".join(s["text"] for s in line["spans"]).strip()
                if not line_text or line_text in ENTETES:
                    continue

                # ── Propriétés typographiques du span dominant ─────────────
                dominant = next((s for s in line["spans"] if s["text"].strip()), None)
                if not dominant:
                    continue
                is_bold = bool(dominant["flags"] & 0b10000)  # bit 4 = bold
                size    = dominant["size"]
                # ── Numéro de page : on le sait toujours (on est dans la boucle page) ──
                last_page_seen = page_num + 1

                # ── Contexte hiérarchique ──────────────────────────────────
                if re.match(r'^LIVRE\b', line_text) and is_bold:
                    current_livre    = line_text
                    current_chapitre = None
                    continue
                if re.match(r'^CHAPITRE\b', line_text) and is_bold:
                    current_chapitre = line_text
                    continue
                # ── Marqueur "Article X"
                art_match = re.match(r'^Article\s+(premier|\d+)$', line_text, re.IGNORECASE)
                if art_match:
                    # Sauvegarder l'article précédent
                    if current_article:
                        # Fusionner les tirets de coupure de colonne (ex: "désa-\nveu")
                        current_article["contenu"] = re.sub(r'-\s+', '', current_article["contenu"]).strip()
                        current_article["titre"]   = re.sub(r'-\s+', '', current_article["titre"]).strip()
                        articles.append(current_article)

                    numero = art_match.group(1)
                    if numero.lower() == "premier":
                        numero = "1"

                    current_article = {
                        "numero":   numero,
                        "titre":    "",
                        "contenu":  "",
                        "page":     last_page_seen,   #  toujours connu ici
                        "livre":    current_livre,
                        "chapitre": current_chapitre,
                        "_attente_titre": True         # flag interne
                    }
                    continue
                # ── Titre de l'article 
                # Le titre = span(s) bold size≈8.5 juste après "Article X" IMPORTANT : le titre peut s'étendre sur PLUSIEURS lignes bold
                if current_article and current_article["_attente_titre"]:
                    if is_bold and size < 9.0:
                        # Accumuler les lignes bold (titre sur 2 lignes)
                        sep = " " if current_article["titre"] else ""
                        current_article["titre"] += sep + line_text
                        continue
                    else:
                        # La ligne n'est plus bold → fin du titre, début du corps
                        current_article["_attente_titre"] = False

                # ── Corps de l'article ─────────────────────────────────────
                if current_article:
                    sep = " " if current_article["contenu"] else ""
                    current_article["contenu"] += sep + line_text
                    
    # Ne pas oublier le dernier article
    if current_article:
        current_article["contenu"] = re.sub(r'-\s+', '', current_article["contenu"]).strip()
        current_article["titre"]   = re.sub(r'-\s+', '', current_article["titre"]).strip()  #  ajout
        articles.append(current_article)

    # Nettoyer le flag interne avant export
    for a in articles:
        a.pop("_attente_titre", None)

    return articles

def export_json(articles, path="code_famille.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f" {len(articles)} articles exportés → {path}")

# ── Pipeline ──────────────────────────────────────────────────────────────────
articles = extract_articles_from_pdf(doc)
export_json(articles)

print(f"\nTotal : {len(articles)} articles\n")
for a in articles[:10]:
    print(json.dumps(a, ensure_ascii=False, indent=2))
    print()