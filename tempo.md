
Comment la photocatalyse rédox est actuellement appliquée en "drug discovery" du secteur pharmaceutique ?


    tk.Label(lbl_container, text="Questions contextuelles :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=5)
    tk.Label(lbl_container, text="Légende :")
    legend_frame = tk.Frame(lbl_container, bg="#323232")
    legend_frame.pack(pady=5, padx=10, anchor="w")
    tk.Label(legend_frame, text="1 mot clef", fg="#ff6b6b", bg="#323232", font=("Segoe UI", 10)).pack(side="left")
    #tk.Label(legend_frame, text="  |  ", fg="white", bg="#323232", font=("Segoe UI", 10)).pack(side="left")
    tk.Label(legend_frame, text="2 ou 3 mots clefs", fg="#ffb347", bg="#323232", font=("Segoe UI", 10)).pack(side="left")
    #tk.Label(legend_frame, text="  |  ", fg="white", bg="#323232", font=("Segoe UI", 10)).pack(side="left")
    tk.Label(legend_frame, text="4 mots clefs ou plus", fg="#599258", bg="#323232", font=("Segoe UI", 10)).pack(side="left")

for text, color in legend_texts:
    tk.Label(
        lbl_container,
        text=text,
        fg=color,
        bg="#323232",
        justify="left",
        wraplength=700,
        font=("Segoe UI", 8, "italic")
    ).pack(anchor="w", padx=20)

# Prints des tuples filtered_context

        def print_tuple_structure(data_tuple, max_depth=2, indent=0):
            prefix = "  " * indent
            if indent > max_depth:
                print(f"{prefix}... (profondeur max atteinte)")
                return

            if isinstance(data_tuple, (tuple, list)):
                print(f"{prefix}Tuple/List (len={len(data_tuple)}):")
                for i, val in enumerate(data_tuple):
                    print(f"{prefix} [{i}] (type={type(val).__name__}): ", end="")
                    if isinstance(val, (tuple, list)):
                        print()
                        print_tuple_structure(val, max_depth=max_depth, indent=indent + 1)
                    else:
                        print(val)
            else:
                print(f"{prefix}{data_tuple} (type={type(data_tuple).__name__})")

        if filtered_context:
            print("Structure du premier élément de filtered_context :")
            print_tuple_structure(filtered_context[0])
        else:
            print("filtered_context est vide.")


    def print_filtered_context_structure(filtered_context, max_items=1):
        print(f"Affichage des {min(len(filtered_context), max_items)} premiers éléments de filtered_context:")
        for i, item in enumerate(filtered_context[:max_items]):
            print(f"Tuple #{i+1} (longueur={len(item)}):")
            for idx, val in enumerate(item):
                print(f"  [{idx}]: {val} (type: {type(val).__name__})")
            print("-" * 40)

    print_filtered_context_structure(filtered_context)

# Code original du notebook avec les vecteurs

def show_infos():
    info_window = tk.Toplevel(root)
    info_window.title("Détails sur le prompt généré")
    info_window.geometry("750x725")
    container = tk.Frame(info_window, bg="#323232")
    container.pack(fill="both", expand=True)

    info_window.transient(root)
    info_window.grab_set()

    question = entry_question.get("1.0", tk.END).strip()
    if not question:
        update_status("⚠️ Posez une question d'abord.", error=True)
        return

    # On récupère le vecteur d'embedding de la question
    question_vector = get_embedding(question)
    if question_vector is None:
        update_status("⚠️ Impossible de générer l'embedding pour la question.", error=True)
        return

    # Récupération des contextes similaires via vecteurs
    filtered_context = get_similar_contexts(question_vector, limit=5)
    if not filtered_context:
        update_status("⚠️ Aucun contexte pertinent trouvé.", error=True)
        return

    # Extraction de mots-clés uniquement pour affichage, mais on ne s'en sert plus pour la recherche
    keywords = extract_keywords(question)
    if not keywords:
        update_status("⚠️ Aucun mot-clé extrait pour affichage.", error=True)
        return

    # --- Onglet Stats ---
    notebook = ttk.Notebook(container, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    single_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(single_tab, text="Mots-clés & Stats")

    # On rassemble le texte complet des contextes pour compter les mots-clés extraits
    full_text_context = " ".join(
        (item[0] + " " + item[1]) if len(item) >= 2 else item[0]
        for item in filtered_context
    ).lower()

    token_list = re.findall(r'\b[a-zA-Z\-]{3,}\b', full_text_context)
    word_counts = {kw: token_list.count(kw.lower()) for kw, _, _ in keywords}

    # Graphique des mots-clés extraits de la question
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    question_keywords = [kw for kw, _, origin in keywords if origin == "question"]
    question_weights = [weight for kw, weight, origin in keywords if origin == "question"]

    ax.bar(question_keywords, question_weights, color='#599258')
    ax.set_facecolor("#323232")
    fig.patch.set_facecolor("#323232")
    ax.set_title("Poids des mots-clés de la question", color="white", fontsize=10)
    ax.set_ylabel("Poids TF-IDF", color="white", fontsize=10)
    ax.tick_params(axis='x', labelrotation=45, colors="white", labelsize=10)
    ax.tick_params(axis='y', colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=single_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
    plt.close(fig)

    # Tableau : mot-clé, poids, fréquence dans contextes
    cols = ('Mot-clé', 'Poids (TF-IDF)', 'Occurrences dans les contextes')
    tree = ttk.Treeview(single_tab, columns=cols, show='headings', height=10, style='Custom.Treeview')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=180)
    tree.pack(expand=False, fill='both', padx=10, pady=(10, 0))

    tree.tag_configure('oddrow', background='#2a2a2a')
    tree.tag_configure('evenrow', background='#383838')

    for i, (kw, weight, _) in enumerate(keywords):
        freq = word_counts.get(kw, 0)
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree.insert('', tk.END, values=(kw, f"{weight:.3f}", freq), tags=(tag,))

    ttk.Button(single_tab, text="Copier mots-clés", command=lambda: (
        root.clipboard_clear(),
        root.clipboard_append(",".join([kw for kw, _, _ in keywords]))
    )).pack(pady=10)

    # --- Onglet contextes ---
    context_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(context_tab, text="Contextes")
    lbl_container = tk.Frame(context_tab, bg="#323232")
    lbl_container.pack(fill="both", expand=True)

    tk.Label(lbl_container, text="Questions contextuelles :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=5)
    tk.Label(lbl_container, text="Légende :\n- 1 mot clef : rouge\n- 2 ou 3 : orange\n- >3 : vert",
             fg="white", bg="#323232", justify="left", wraplength=700, font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=10, pady=(0, 10))

    base_keywords = set(kw for kw, _, _ in keywords)
    for item in filtered_context:
        q_text = item[0]
        extracted = set(kw for kw, _, _ in extract_keywords(q_text))
        shared = len(extracted & base_keywords)
        color = "#ff6b6b" if shared <= 1 else "#ffb347" if shared <= 3 else "#7CFC00"
        tk.Label(lbl_container, text=q_text[:100] + ("..." if len(q_text) > 100 else ""), fg=color, bg="#323232", wraplength=700).pack(anchor="w", padx=10)

    # --- Onglet carte mentale ---
    mindmap_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(mindmap_tab, text="Carte mentale")

    canvas = tk.Canvas(mindmap_tab, bg="#323232", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    import math
    questions_list = [item[0] for item in filtered_context]
    keywords_per_question = [set(kw for kw, _, _ in extract_keywords(q)) for q in questions_list]
    center_x, center_y, radius = 350, 300, 250
    radius_node = 15
    positions = [(center_x + radius * math.cos(2 * math.pi * i / len(questions_list)),
                  center_y + radius * math.sin(2 * math.pi * i / len(questions_list)))
                 for i in range(len(questions_list))]

    for i in range(len(questions_list)):
        for j in range(i + 1, len(questions_list)):
            if keywords_per_question[i] & keywords_per_question[j]:
                canvas.create_line(*positions[i], *positions[j], fill="#7CFC00", width=1)

    tooltip = None

    def show_tooltip(event, text):
        nonlocal tooltip
        if tooltip:
            canvas.delete(tooltip)
        tooltip = canvas.create_text(event.x + 10, event.y + 10, text=text, anchor="nw", fill="white", font=("Segoe UI", 9), width=300)

    def hide_tooltip(event):
        nonlocal tooltip
        if tooltip:
            canvas.delete(tooltip)
            tooltip = None

    for i, (x, y) in enumerate(positions):
        node = canvas.create_oval(x - radius_node, y - radius_node, x + radius_node, y + radius_node, fill="#599258", outline="")
        canvas.tag_bind(node, "<Enter>", lambda e, q=questions_list[i]: show_tooltip(e, q))
        canvas.tag_bind(node, "<Leave>", hide_tooltip)

    # --- Onglet Stats Globales ---
    global_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(global_tab, text="Stats Globales")

    top_n = 20
    top_keywords = global_keywords_stats[:top_n]
    labels = [kw for kw, _, _ in top_keywords]
    occurrences = [count for _, count, _ in top_keywords]

    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=100)
    ax2.bar(labels, occurrences, color='#457b9d')
    ax2.set_facecolor("#323232")
    fig2.patch.set_facecolor("#323232")
    ax2.set_title("Top 20 mots-clés de toute la base", color="white", fontsize=10)
    ax2.set_ylabel("Occurrences totales", color="white", fontsize=10)
    ax2.tick_params(axis='x', labelrotation=45, colors="white", labelsize=10)
    ax2.tick_params(axis='y', colors="white", labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color('white')

    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=global_tab)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
    plt.close(fig2)

    search_var = tk.StringVar()

    def filter_treeview(*args):
        search_term = search_var.get().lower()
        tree_global.delete(*tree_global.get_children())
        for i, (kw, total, nb_conv) in enumerate(global_keywords_stats):
            if search_term in kw.lower():
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                tree_global.insert('', tk.END, values=(kw, total, nb_conv), tags=(tag,))

    search_var.trace_add('write', filter_treeview)

    search_entry = ttk.Entry(global_tab, textvariable=search_var)
    search_entry.pack(fill='x', padx=10, pady=(10, 0))
    search_entry.insert(0, "Rechercher un mot-clé...")

    def on_entry_click(event):
        if search_entry.get() == "Rechercher un mot-clé...":
            search_entry.delete(0, tk.END)
    search_entry.bind('<FocusIn>', on_entry_click)

    cols_global = ('Mot-clé', 'Occurrences totales', 'Nb conversations')
    tree_global = ttk.Treeview(global_tab, columns=cols_global, show='headings', height=15, style='Custom.Treeview')
    for col in cols_global:
        tree_global.heading(col, text=col)
        tree_global.column(col, width=180)
    tree_global.pack(expand=False, fill='both', padx=10, pady=(5, 0))

    tree_global.tag_configure('oddrow', background='#2a2a2a')
    tree_global.tag_configure('evenrow', background='#383838')

    for i, (kw, total, nb_conv) in enumerate(global_keywords_stats):
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree_global.insert('', tk.END, values=(kw, total, nb_conv), tags=(tag,))


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