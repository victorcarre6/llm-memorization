
Comment la photocatalyse rédox est actuellement appliquée en "drug discovery" du secteur pharmaceutique ?



# Bouton pour étendre/cacher
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


# === CONSTRUCTION DU PROMPT ===

# Compression du contexte extrait
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
    
# Construction du prompt
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