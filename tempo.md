
Comment la photocatalyse rédox est actuellement appliquée en "drug discovery" du secteur pharmaceutique ?
# remplacé par nouveau histogramme (à partir de all_keyword)


    # Calcul des fréquences dans les contextes
    full_text_context = " ".join(user_input + " " + llm_output for user_input, llm_output, _, _ in filtered_context).lower()
    token_list = re.findall(r'\b[a-zA-Z\-]{3,}\b', full_text_context)

    word_counts = {}
    for kw, _, _ in keywords:
        count = token_list.count(kw.lower())
        word_counts[kw] = count


    # --- Histogramme ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    bars = ax.bar(word_counts.keys(), word_counts.values(), color='#599258')

    ax.set_facecolor("#323232")
    fig.patch.set_facecolor("#323232")
    ax.set_title("Fréquence des mots-clés dans les contextes", color="white", fontsize=10)
    ax.set_ylabel("Occurrences", color="white", fontsize=10)
    ax.tick_params(axis='x', labelrotation=45, colors="white", labelsize=10)
    ax.tick_params(axis='y', colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=single_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))





#

def get_relevant_context(user_question, limit=None):
    if limit is None:
        limit = context_count_var.get()  # Récupération dynamique si pas de limite donnée
    keywords = extract_keywords(user_question)
    if not keywords:
        return []
    # Préparation de la requête SQL
    placeholders = ', '.join(['?'] * len(keywords))
    query = f'''
        SELECT c.id, c.user_input, c.llm_output, c.timestamp, k.keyword
        FROM conversations c
        JOIN keywords k ON c.id = k.conversation_id
        WHERE k.keyword IN ({placeholders})
    '''


    keyword_strings = [kw[0] for kw in keywords]
    cur.execute(query, keyword_strings)
    rows = cur.fetchall()
    match_counts = {}
    context_data = {}
    for convo_id, user_input, llm_output, timestamp, keyword in rows:
        if convo_id not in match_counts:
            match_counts[convo_id] = set()
            context_data[convo_id] = (user_input, llm_output, timestamp)
        match_counts[convo_id].add(keyword)
    scored_contexts = [(convo_id, len(matched_keywords)) for convo_id, matched_keywords in match_counts.items()] # Calcul du score (nombre de mots-clés distincts en commun)
    sorted_contexts = sorted(scored_contexts, key=lambda x: x[1], reverse=True) # Tri décroissant par score
    filtered_context = [context_data[convo_id] for convo_id, score in sorted_contexts[:limit]]  # Sélection des meilleurs résultats selon la limite
    return filtered_context

# Old

## Bouton pour étendre/cacher
toggle_btn = ttk.Button(
    output_frame,
    text="▼ Afficher les résultats",
    command=toggle_output,
    bg="#599258",
    fg="white",
    font=('Segoe UI', 9),
    relief=tk.FLAT,
    pady=4,
    cursor="hand2",
    activebackground="#457a3a"
)
toggle_btn.pack(fill=tk.X, pady=(0, 5))


## === CONSTRUCTION DU PROMPT ===

### Compression du contexte extrait
def summarize(text, focus_terms=None, max_length=1024):
    transformers_logging.set_verbosity_error()
    try:
        # Filtrage des phrases importantes si focus_terms donné
        if focus_terms:
            sentences = [s for s in text.split('.') 
                        if any(term.lower() in s.lower() for term in focus_terms)]
            text = '. '.join(sentences)[:2000] or text[:2000]

        # Résumé avec le texte filtré
        result = summarizer(
            text,
            max_length=max_length,
            min_length=max_length // 2,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )
        return nlp_clean_text(result[0]['summary_text'])

    except Exception as e:
        print(f"Erreur summarization : {e}")
        return text[:max_length] + "... [résumé tronqué]"
    
### Construction du prompt
def generate_prompt_paragraph(context, question, target_tokens=1000):
    if not context:
        return f"{question}"

    # 1. Prétraitement
    processed_items = []
    for item in context[:3]:  # Nombre max d'éléments dans le contexte
        try:
            # Extraction sécurisée
            user_input = str(item[0])[:300]  # Troncature des questions longues
            llm_output = str(item[1])
            keyword = str(item[5]) if len(item) > 5 and str(item[3]).strip() not in {"", "none", "null", "1", "2", "3"} else None

            # Summarization
            summary = summarize(text=llm_output)

            # Nettoyage et segmentation du texte via nlp_clean_text
            cleaned_summary = nlp_clean_text(summary)
            processed_items.append({
                'question': user_input,
                'summary': cleaned_summary,
                'keyword': keyword.lower().strip() if keyword else None
            })

        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue