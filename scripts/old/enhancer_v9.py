import sqlite3
import re
import pyperclip
import tkinter as tk
import subprocess
import json
import sklearn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tkinter import scrolledtext, ttk, Canvas
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from collections import Counter
import os
import webbrowser
import spacy
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === INITIALISATION ===

# Chemin absolu vers le dossier racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(PROJECT_ROOT, "config.json")
with open(config_path, "r") as f:
    raw_config = json.load(f)

# Chargement de la config
config = {}
for key, value in raw_config.items():
    if isinstance(value, str):
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            expanded = os.path.normpath(os.path.join(PROJECT_ROOT, expanded))
        config[key] = expanded
    else:
        config[key] = value
stopwords_path = config.get("stopwords_file_path", "stopwords_fr.json")
with open(stopwords_path, "r", encoding="utf-8") as f:
    french_stop_words = set(json.load(f))

# Modèles

summarizing_model = "moussaKam/barthez-orangesum-abstract"

# Connexion à la base SQLite
conn = sqlite3.connect(config["db_path"])
cur = conn.cursor()

# Initialisation des modèles
nlp = spacy.load("fr_core_news_lg")
kw_model = KeyBERT()
model = AutoModelForSeq2SeqLM.from_pretrained(summarizing_model)
tokenizer = AutoTokenizer.from_pretrained(summarizing_model)
summarizer = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer,
    framework="pt"
)

# === FONCTIONS PRINCIPALES ===

# Pércuteur

def on_ask():
    question = entry_question.get("1.0", "end-1c")
    if not question.strip():
        update_status("⚠️ Merci de saisir une question.", error=True)
        return
    
    update_status("Traitement en cours...")
    root.update()

    try:
        context = get_relevant_context(question, limit=context_count_var.get()) #", limit=context_count_var.get()" ajoutée slider contexte
        prompt = generate_prompt_paragraph(context, question)
        pyperclip.copy(prompt)

        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, prompt)
        
        # Calcul des métriques
        context_count = len(context)
        token_count = len(prompt.split())
        
        update_status(
            f"Prompt généré ({token_count} tokens) | Contexte utilisé : {context_count} éléments",
            success=True
        )
        
    except Exception as e:
        update_status(f"Erreur : {str(e)}", error=True)

def update_status(message, error=False, success=False):
    """Met à jour le label de statut avec style approprié"""
    label_status.config(text=message)
    if error:
        label_status.config(foreground='#ff6b6b')
    elif success:
        label_status.config(foreground='#599258')
    else:
        label_status.config(foreground='white')

# Synchronisation des conversations
def sync_conversations():
    try:
        global config
        sync_path = config.get("sync_script_path")
        if not sync_path:
            label_status.config(text="sync_script_path introuvable.")
            return

        subprocess.run(["python3", config["sync_script_path"]], check=True)
        label_status.config(text="Synchronisation terminée.")
    except subprocess.CalledProcessError:
        label_status.config(text="Erreur lors de la synchronisation.")


# === RÉCUPÉRATION DU CONTEXTE ===

# Récupération des mots-clés de la question initiale

root = tk.Tk()
keyword_count_var = tk.IntVar(value=5)  # Valeur par défaut
context_count_var = tk.IntVar(value=3)  # Valeur par défaut du nombre de contextes

def extract_keywords(text, top_n=None):
    if top_n is None:
        top_n = keyword_count_var.get()

    # Extraction brute avec KeyBERT
    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),  # mots uniques uniquement
        stop_words='english',
        top_n=top_n * 2  # extraire plus pour filtrer ensuite
    )

    tokens = re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower())
    token_freq = Counter([tok for tok in tokens if tok not in french_stop_words])

    # Tri par fréquence dans le texte #  /!\ peu entrainer des doublons /!\
    raw_keywords_sorted = sorted(
        raw_keywords,
        key=lambda x: token_freq.get(x[0].lower(), 0),
        reverse=True
    )

    seen = set()
    filtered_keywords = []

    for kw, weight in raw_keywords_sorted:
        kw_clean = kw.lower().strip()
        if (
            kw_clean not in seen and
            kw_clean not in french_stop_words and
            len(kw_clean) > 2 and
            re.match(r'^[a-zA-Z\-]+$', kw_clean)
        ):
            seen.add(kw_clean)
            freq = token_freq.get(kw_clean, 0)
            filtered_keywords.append((kw_clean, weight, freq))

        if len(filtered_keywords) >= top_n:
            break

    return filtered_keywords

# Requête SQL en fonction des mots-clés extraits
def get_relevant_context(user_question, limit=None):
    if limit is None:
        limit = context_count_var.get()  # Récupération dynamique si pas de limite donnée
    
    keywords = extract_keywords(user_question)
    print(f"Mots-clés extraits de la question : {keywords}")
    if not keywords:
        # Pas de mots-clés extraits, on retourne une liste vide
        return []

    # Préparation de la requête SQL avec placeholders pour les mots-clés
    placeholders = ', '.join(['?'] * len(keywords))
    query = f'''
        SELECT c.id, c.user_input, c.llm_output, c.timestamp, k.keyword
        FROM conversations c
        JOIN keywords k ON c.id = k.conversation_id
        WHERE k.keyword IN ({placeholders})
    '''
    keyword_strings = [kw[0] for kw in keywords]  # extraire juste les mots
    cur.execute(query, keyword_strings)
    rows = cur.fetchall()

    # Dictionnaires pour compter les mots-clés en commun par conversation
    match_counts = {}
    context_data = {}

    for convo_id, user_input, llm_output, timestamp, keyword in rows:
        if convo_id not in match_counts:
            match_counts[convo_id] = set()
            context_data[convo_id] = (user_input, llm_output, timestamp)
        match_counts[convo_id].add(keyword)

    scored_contexts = [(convo_id, len(matched_keywords)) for convo_id, matched_keywords in match_counts.items()] # Calcul du score = nombre de mots-clés distincts en commun
    sorted_contexts = sorted(scored_contexts, key=lambda x: x[1], reverse=True) # Tri décroissant par score (nombre de mots-clés communs)
    filtered_context = [context_data[convo_id] for convo_id, score in sorted_contexts[:limit]]  # Sélection des meilleurs résultats selon la limite

    return filtered_context


# === FONCTION UTILITAIRE NLP  ===

def nlp_clean_text(text, max_chunk_size=500):
    # Suppression des blocs de code
    text = re.sub(r'```(?:python)?\s*.*?```', '', text, flags=re.DOTALL)

    # Analyse NLP
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 20:
            continue  # ignore phrases trop courtes

        if current_length + len(s) < max_chunk_size:
            current_chunk.append(s)
            current_length += len(s)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_length = len(s)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return " ".join(chunks[:3])  # limite à 3 blocs maximum

# === CONSTRUCTION DU PROMPT ===

# Compression primaire du contexte extrait
def summarize(text, focus_terms=None, max_length=1024):
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
    question_keywords = extract_keywords(question) #KBogus enlevé "", top_n=keyword_count_var.get()" de la parenthèse
    if not context:
        return f"Voici une nouvelle question à traiter : {question}"

    # 1. Prétraitement intelligent du contexte
    processed_items = []
    for item in context[:3]:  # Nombre max d'éléments dans le contexte
        try:
            # Extraction sécurisée
            user_input = str(item[0])[:300]  # Troncature des questions longues
            llm_output = str(item[1])
            keyword = str(item[5]) if len(item) > 5 and str(item[3]).strip() not in {"", "none", "null", "1", "2", "3"} else None

            # Summarization avec gestion de la longueur
            summary = summarize(text=llm_output)

            # Nettoyage et segmentation du texte via nlp_clean_text
            print(f"Avant segmentation : {len(summary.split())} mots")
            cleaned_summary = nlp_clean_text(summary)
            processed_items.append({
                'question': user_input,
                'summary': cleaned_summary,
                'keyword': keyword.lower().strip() if keyword else None
            })

        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue

    # 2. Construction du prompt
    parts = []

    # Partie questions
    if processed_items:
        questions = [f"'{item['question']}'" for item in processed_items]
        if len(questions) == 1:
            parts.append(f"Question précédente : {questions[0]}")
        else:
            *init, last = questions
            parts.append(f"Questions antérieures : {', '.join(init)}, et enfin {last}")

    # Partie mots-clés
    keywords = {item['keyword'] for item in processed_items if item['keyword']}
    if keywords:
        parts.append(f"Mots-clés pertinents : {', '.join(sorted(keywords))}")

    # Partie résumés
    if processed_items:
        summaries = [f"- {item['summary']}" for item in processed_items]
        parts.append("Contexte pertinent :\n" + "\n".join(summaries))

    # Question actuelle
    parts.append(f"\nQuestion à traiter : {question}")

    return "\n".join(parts)

# === INTERFACE TKINTER ===

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")

def show_help():
    help_text = (
        "LLM Memorization and Prompt Enhancer — Aide\n\n"
        "• Synchroniser les conversations : ajoute les nouveaux échanges depuis LM Studio à la base de données.\n\n"
        "• Générer prompt : extrait les mots-clés de votre question, cherche des conversations similaires dans votre base SQL, puis compresse les informations avec un LLM local.\n\n"
        "Le prompt final est affiché puis automatiquement copié dans votre presse-papier !\n\n"
        "Pour en savoir plus, obtenir plus d'informations à propos d'un éventuel bloquage des scripts, ou me contacter, visitez : github.com/victorcarre6/llm-memorization"
    )
    help_window = tk.Toplevel(root)
    help_window.title("Aide")
    help_window.geometry("500x300")
    help_window.configure(bg="#323232")

    text_widget = tk.Text(help_window, wrap=tk.WORD, font=("Segoe UI", 8))
    text_widget.insert(tk.END, help_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    ok_button = tk.Button(help_window, text="Fermer", command=help_window.destroy)
    ok_button.pack(pady=10)

# === CONFIGURATION DE L'INTERFACE ===
root.title("LLM Memorization and Prompt Enhancer")
root.geometry("1000x750")
root.configure(bg="#323232")

# Style global unique
style = ttk.Style(root)
style.theme_use('clam')

# Configuration du style
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 11),
        'padding': 6
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 9),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 8, 'italic underline'),
        'padding': 0
    },
    'TLabel': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 11)
    },
    'TEntry': {
        'fieldbackground': '#FDF6EE',
        'foreground': 'black',
        'font': ('Segoe UI', 11)
    },
    'TFrame': {
        'background': '#323232'
    },
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 11)
    },
    'TNotebook': {
        'background': '#323232',
        'borderwidth': 0
    },
    'TNotebook.Tab': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'padding': (10, 4)
    },
    'Custom.Treeview': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'fieldbackground': '#2a2a2a',
        'font': ('Segoe UI', 10),
        'bordercolor': '#323232',
        'borderwidth': 0,
    },
    'Custom.Treeview.Heading': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 11, 'bold'),
        'relief': 'flat'
    }
}


for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map("TNotebook.Tab",
          background=[("selected", "#323232"), ("active", "#2a2a2a")],
          foreground=[("selected", "white"), ("active", "white")])


# Widgets principaux
main_frame = ttk.Frame(root, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Section question - Centrée
question_header = ttk.Frame(main_frame, style='TFrame')
question_header.pack(fill='x', pady=(0, 5))
ttk.Label(question_header, text="Poser la question :").pack(expand=True)

question_frame = tk.Frame(main_frame, bg="#323232")
question_frame.pack(pady=(0, 10), fill='x', expand=True)

entry_question = tk.Text(question_frame, height=4, width=80, wrap="word", font=('Segoe UI', 11))
entry_question.pack(side="left", fill="both", expand=True)

# Configuration du style pour la scrollbar
style = ttk.Style()
style.configure("Vertical.TScrollbar",
    troughcolor='#FDF6EE',
    background='#C0C0C0',
    darkcolor='#C0C0C0',
    lightcolor='#C0C0C0',
    bordercolor='#FDF6EE',
    arrowcolor='black',
    relief='flat')

scrollbar = ttk.Scrollbar(
    question_frame,
    orient="vertical",
    command=entry_question.yview,
    style="Vertical.TScrollbar"  # Application du style
)
scrollbar.pack(side="right", fill="y")

entry_question.config(yscrollcommand=scrollbar.set)
entry_question.bind("<Return>", lambda event: on_ask())

# Frame horizontale principale
control_frame = ttk.Frame(main_frame, style='TFrame')
control_frame.pack(fill='x', pady=(0, 15), padx=5)

# Colonne 1 : slider mots-clés
slider_keywords_frame = ttk.Frame(control_frame, style='TFrame')
slider_keywords_frame.grid(row=0, column=0, sticky='w')

label_keyword_count = ttk.Label(slider_keywords_frame, text=f"Nombre de mots-clés : {keyword_count_var.get()}", style='TLabel')
label_keyword_count.pack(anchor='w')

slider_keywords = ttk.Scale(
    slider_keywords_frame,
    from_=1,
    to=15,
    orient="horizontal",
    variable=keyword_count_var,
    length=180,
    command=lambda val: label_keyword_count.config(text=f"Nombre de mots-clés : {int(float(val))}")
)
slider_keywords.pack(anchor='w')

# Colonne 2 : slider contextes + label dynamique au-dessus
slider_context_frame = ttk.Frame(control_frame, style='TFrame')
slider_context_frame.grid(row=0, column=1, padx=20, sticky='w')

label_contexts_count = ttk.Label(slider_context_frame, text=f"Nombre de contextes : {context_count_var.get()}", style='TLabel')
label_contexts_count.pack(anchor='w')

slider_contexts = ttk.Scale(
    slider_context_frame,
    from_=1,
    to=5,
    orient=tk.HORIZONTAL,
    variable=context_count_var,
    length=170,
    command=lambda val: label_contexts_count.config(text=f"Nombre de contextes : {int(float(val))}")
)
slider_contexts.pack(anchor='w')


# Colonne 3 : boutons alignés à droite
button_frame = ttk.Frame(control_frame, style='TFrame')
button_frame.grid(row=0, column=2, sticky='e')

sync_button = ttk.Button(button_frame, text="Synchroniser les conversations", 
                         command=sync_conversations, style='Green.TButton')
sync_button.pack(side='left', padx=5)

btn_ask = ttk.Button(button_frame, text="Générer prompt", command=on_ask, style='Green.TButton')
btn_ask.pack(side='left', padx=5)
control_frame.grid_columnconfigure(2, weight=1)

# Zone de sortie
output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

text_output = scrolledtext.ScrolledText(
    output_frame, 
    width=100, 
    height=20,
    font=('Segoe UI', 11), 
    wrap=tk.WORD, 
    bg="#FDF6EE", 
    fg="black", 
    insertbackground="black"
)
text_output.pack(fill=tk.BOTH, expand=True)

# Barre de statut
status_frame = ttk.Frame(main_frame, style='TFrame')
status_frame.pack(fill=tk.X, pady=(0, 5))

label_status = ttk.Label(
    status_frame, 
    text="Prêt", 
    style='Status.TLabel',
    foreground='white',
    anchor='center',
    justify='center',
    wraplength=650
)
label_status.pack(side=tk.LEFT)


def show_infos():
    info_window = tk.Toplevel(root)
    info_window.title("Détails sur le prompt généré")
    info_window.geometry("750x650")
    container = tk.Frame(info_window, bg="#323232")
    container.pack(fill="both", expand=True)

    info_window.transient(root)
    info_window.grab_set()

    question = entry_question.get("1.0", tk.END).strip()
    if not question:
        update_status("⚠️ Posez une question d'abord pour accéder aux infos.", error=True)
        info_window.destroy()
        return

    keywords = extract_keywords(question)  # [(kw, weight, freq), ...]
    context = get_relevant_context(question)  # Liste de tuples (user_input, llm_output, timestamp)

    notebook = ttk.Notebook(container, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    single_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(single_tab, text="Mots-clés & Stats")

    # Calcul des fréquences dans les contextes
    full_text_context = " ".join(user_input + " " + llm_output for user_input, llm_output, _ in context).lower()
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

    # --- Tableau Treeview ---
    cols = ('Mot-clé', 'Poids', 'Occurrences dans contextes')
    tree = ttk.Treeview(single_tab, columns=cols, show='headings', height=10, style='Custom.Treeview')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=150)
    tree.pack(expand=False, fill='both', padx=10, pady=(10, 0))

    # Alternance des lignes
    tree.tag_configure('oddrow', background='#2a2a2a')
    tree.tag_configure('evenrow', background='#383838')

    for i, (kw, weight, _) in enumerate(keywords):
        freq_in_context = word_counts.get(kw, 0)
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree.insert('', tk.END, values=(kw, f"{weight:.3f}", freq_in_context), tags=(tag,))

    # --- Bouton copier ---
    btn_copy = ttk.Button(single_tab, text="Copier mots-clés", command=lambda: (
        root.clipboard_clear(),
        root.clipboard_append(",".join([kw for kw, _, _ in keywords])),
        print("Mots-clés copiés:", ",".join([kw for kw, _, _ in keywords]))
    ), style='Green.TButton')
    btn_copy.pack(pady=10)



    # === Onglet 2 : Contexte ===
    context_frame = ttk.Frame(notebook, style="TFrame")
    notebook.add(context_frame, text="Contextes")

    # Utiliser Frame classique pour labels avec bg
    lbl_container = tk.Frame(context_frame, bg="#323232")
    lbl_container.pack(fill="both", expand=True)

    tk.Label(lbl_container, text="Questions contextuelles :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=5)
    tk.Label(lbl_container,
             text="Légende :\n- 1 mot clef : rouge\n- 2 ou 3 : orange\n- >3 : vert",
             fg="white", bg="#323232", justify="left", wraplength=700, font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=10, pady=(0, 10))

    for item in context:
        q_text = item[0]
        extracted = extract_keywords(q_text)
        shared = len(set(extracted) & set(keywords))
        color = "#ff6b6b" if shared <= 1 else "#ffb347" if shared <= 3 else "#7CFC00"
        tk.Label(
            lbl_container,
            text=q_text[:100] + ("..." if len(q_text) > 100 else ""),
            fg=color, bg="#323232", wraplength=700
        ).pack(anchor="w", padx=10)


    # === Onglet 3 : Carte mentale ===
    mindmap_frame = ttk.Frame(notebook, style="TFrame")
    notebook.add(mindmap_frame, text="Carte mentale")

    canvas = tk.Canvas(mindmap_frame, bg="#323232", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Préparer données pour carte mentale
    questions_list = [item[0] for item in context]  # liste des questions contextuelles
    keywords_per_question = [extract_keywords(q) for q in questions_list]

    import math
    n = len(questions_list)
    center_x, center_y = 350, 300
    radius = 250
    nodes_positions = []

    # Positionner les nœuds en cercle
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        nodes_positions.append((x, y))

    # Dessiner les liens entre nœuds
    for i in range(n):
        for j in range(i+1, n):
            if set(keywords_per_question[i]) & set(keywords_per_question[j]):
                x1, y1 = nodes_positions[i]
                x2, y2 = nodes_positions[j]
                canvas.create_line(x1, y1, x2, y2, fill="#7CFC00", width=1)

    # Dessiner les nœuds (cercles)
    radius_node = 15

    _tooltip = None
    def show_tooltip(event, text):
        nonlocal _tooltip
        if _tooltip:
            canvas.delete(_tooltip)
            _tooltip = None
        x, y = event.x + 10, event.y + 10
        _tooltip = canvas.create_text(x, y, text=text, anchor="nw", fill="white", font=("Segoe UI", 9), width=300, tags="tooltip")

    def hide_tooltip(event):
        nonlocal _tooltip
        if _tooltip:
            canvas.delete(_tooltip)
            _tooltip = None

    for i, (x, y) in enumerate(nodes_positions):
        node = canvas.create_oval(x-radius_node, y-radius_node, x+radius_node, y+radius_node, fill="#599258", outline="")
        # Bind survol pour tooltip
        canvas.tag_bind(node, "<Enter>", lambda e, q=questions_list[i]: show_tooltip(e, q))
        canvas.tag_bind(node, "<Leave>", hide_tooltip)


# Boutons droite
right_buttons = ttk.Frame(main_frame)
right_buttons.pack(side=tk.RIGHT, anchor='ne', pady=(10, 0))

btn_info = ttk.Button(right_buttons, text="Détails", style='Bottom.TButton', command=show_infos, width=8)
btn_info.pack(pady=(0, 3))
btn_help = ttk.Button(right_buttons, text="Aide", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack()

# Footer
footer_frame = ttk.Frame(root, style='TFrame')
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

dev_label = ttk.Label(footer_frame, text="Développé par Victor Carré —", style='TLabel', font=('Segoe UI', 8))
dev_label.pack(side=tk.LEFT)

github_link = ttk.Label(footer_frame, text="GitHub", style='Blue.TLabel', cursor="hand2")
github_link.pack(side=tk.LEFT)
github_link.bind("<Button-1>", open_github)


root.mainloop()