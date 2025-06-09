import sqlite3
import re
import pyperclip
import tkinter as tk
import subprocess
import json
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tkinter import scrolledtext, ttk
from transformers import pipeline, AutoTokenizer
from keybert import KeyBERT
import os
import webbrowser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Chemin absolu vers le dossier racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

config_path = os.path.join(PROJECT_ROOT, "config.json")
with open(config_path, "r") as f:
    raw_config = json.load(f)

config = {}
for key, value in raw_config.items():
    if isinstance(value, str):
        expanded = os.path.expanduser(value)
        if not os.path.isabs(expanded):
            expanded = os.path.normpath(os.path.join(PROJECT_ROOT, expanded))
        config[key] = expanded
    else:
        config[key] = value

# --- Constantes ---
TOP_K = 5

# --- Connexion à la base ---
conn = sqlite3.connect(config["db_path"])
cur = conn.cursor()

# --- Initialisation des modèles ---
kw_model = KeyBERT()  # modèle d'extraction de mots-clés local
summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    device="mps",  # Pour Apple Silicon !!
    tokenizer=AutoTokenizer.from_pretrained("Falconsai/text_summarization")
)

# === FONCTIONS ===

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

def extract_keywords(text, top_n=20):
    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),  # mots uniques uniquement
        stop_words='english',
        top_n=top_n * 2  # extraire plus pour filtrer ensuite
    )

    seen = set()
    filtered_keywords = []

    for kw, _ in raw_keywords:
        kw_clean = kw.lower().strip()
        # filtre : pas de stopwords, que des mots alphabétiques, min 3 lettres, pas de doublons
        if (
            kw_clean not in seen and
            kw_clean not in ENGLISH_STOP_WORDS and
            len(kw_clean) > 2 and
            re.match(r'^[a-zA-Z\-]+$', kw_clean)
        ):
            seen.add(kw_clean)
            filtered_keywords.append(kw_clean)

        if len(filtered_keywords) >= top_n:
            break

    return filtered_keywords

def get_relevant_context(user_question, limit=TOP_K):
    keywords = extract_keywords(user_question)
    print(f"Mots-clés extraits de la question : {keywords}")
    if not keywords:
        return []

    # Récupérer toutes les conversations contenant au moins un mot-clé
    placeholders = ', '.join(['?'] * len(keywords))
    query = f'''
        SELECT c.id, c.user_input, c.llm_output, c.timestamp, k.keyword
        FROM conversations c
        JOIN keywords k ON c.id = k.conversation_id
        WHERE k.keyword IN ({placeholders})
    '''
    cur.execute(query, (*keywords,))
    rows = cur.fetchall()

    # Compter les mots-clés en commun par conversation
    match_counts = {}
    context_data = {}

    for convo_id, user_input, llm_output, timestamp, keyword in rows:
        if convo_id not in match_counts:
            match_counts[convo_id] = set()
            context_data[convo_id] = (user_input, llm_output, timestamp)
        match_counts[convo_id].add(keyword)

    # Calcul du score = nombre de mots-clés en commun
    scored_contexts = [
        (convo_id, len(keywords_matched))
        for convo_id, keywords_matched in match_counts.items()
    ]

    # Tri décroissant selon le nombre de mots-clés en commun
    sorted_contexts = sorted(scored_contexts, key=lambda x: x[1], reverse=True)

    # Récupération des résultats top-N
    filtered_context = []
    for convo_id, score in sorted_contexts[:limit]:
        filtered_context.append(context_data[convo_id])

    return filtered_context


def clean_text(text):
    # Supprime les blocs de code délimités par ```python``` et ```
    cleaned = re.sub(r'```python.*?```', '', text, flags=re.DOTALL)
    return cleaned.strip()

def summarize_text(text):
    cleaned = clean_text(text)
    summary = summarizer(cleaned, do_sample=False)
    return summary[0]['summary_text']

# Fonctions utilitaires
def clean_summary(text):
    """Nettoie le texte pour un français correct"""
    text = text.lower()
    text = text.replace("and", "et")
    text = text.replace(":", ",")
    text = text.replace("  ", " ")
    if not text.endswith((".", "!", "?")):
        text = text + "."
    return text.capitalize()

def clean_french(text):
    """Corrige les erreurs de français courantes"""
    replacements = [
        ("le processus", "la procédure"),
        ("testing", "test"),
        ("identifying", "identification"),
        ("synthésising", "synthèse"),
        ("drug discovery", "découverte de médicaments")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def generate_prompt_paragraph(context, question, target_tokens=1000):
    """Version finale avec gestion robuste des longs textes et optimisation MPS"""

    question_keywords = extract_keywords(question, top_n=5)
    if not context:
        return f"Voici une nouvelle question à traiter : {question}"

    # 1. Prétraitement intelligent du contexte
    processed_items = []
    for item in context[:3]:  # Limiter à 3 éléments max
        try:
            # Extraction sécurisée
            user_input = str(item[0])[:150]  # Troncature des questions longues
            llm_output = str(item[1])
            keyword = str(item[3]) if len(item) > 3 and str(item[3]).strip() not in {"", "none", "null", "1", "2", "3"} else None
            
            # Nettoyage et segmentation du texte
            cleaned_text = clean_and_segment_text(llm_output)
            
            # Summarization avec gestion de la longueur
            summary = safe_summarize(
                text=cleaned_text,
                focus_terms=question_keywords,  # Utilisation directe des keywords
                max_length=120
            )
            
            processed_items.append({
                'question': user_input,
                'summary': summary,
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
    
    return "\n\n".join(parts)

def clean_and_segment_text(text, max_chunk_size=500):
    """Découpe le texte en segments pertinents"""
    # 1. Nettoyage de base
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. Découpage par phrases techniques
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) < max_chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return ' '.join(chunks[:3])  # Limite à 3 chunks max

def safe_summarize(text, focus_terms=None, max_length=120):
    """Summarization robuste pour Apple Silicon"""
    try:
        # Pré-filtrage des phrases importantes
        if focus_terms:
            sentences = [s for s in text.split('.') 
                        if any(term.lower() in s.lower() for term in focus_terms)]
            text = '. '.join(sentences)[:2000] or text[:2000]
        
        # Paramètres optimisés pour MPS
        result = summarizer(
            text,
            max_length=max_length,
            min_length=max_length//2,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )
        return clean_french(result[0]['summary_text'])
    
    except Exception as e:
        print(f"Erreur summarization : {e}")
        return text[:max_length] + "... [résumé tronqué]"


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

# === INTERFACE TKINTER ===

def on_ask():
    question = entry_question.get()
    if not question.strip():
        update_status("⚠️ Merci de saisir une question.", error=True)
        return
    
    update_status("⌛ Traitement en cours...")
    root.update()

    try:
        context = get_relevant_context(question)
        prompt = generate_prompt_paragraph(context, question)
        pyperclip.copy(prompt)

        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, prompt)
        
        # Calcul des métriques
        context_count = len(context)
        token_count = len(prompt.split())  # Estimation simple - remplacer par tokenizer réel si nécessaire
        
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

# Configuration de l'interface
root = tk.Tk()
root.title("LLM Memorization and Prompt Enhancer")
root.geometry("850x650")  # Légèrement augmenté pour meilleure disposition
root.configure(bg="#323232")

# Style
style = ttk.Style(root)
style.theme_use('clam')

# Configuration des styles
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 11),
        'padding': 6
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
    'TFrame': {'background': '#323232'},
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 11)

    }
}

for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

# Widgets principaux
main_frame = ttk.Frame(root, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Section question
ttk.Label(main_frame, text="Poser la question :").pack(pady=(0, 5))
entry_question = ttk.Entry(main_frame, width=80, style='TEntry')
entry_question.pack(pady=(0, 10))
entry_question.bind("<Return>", lambda event: on_ask())

# Boutons
button_frame = ttk.Frame(main_frame, style='TFrame')
button_frame.pack(pady=(0, 10))

sync_button = ttk.Button(button_frame, text="Synchroniser les conversations", 
                        command=sync_conversations, style='Green.TButton')
sync_button.pack(side=tk.LEFT, padx=5)

btn_ask = ttk.Button(button_frame, text="Générer prompt", command=on_ask, style='Green.TButton')
btn_ask.pack(side=tk.LEFT, padx=5)

# Zone de sortie
output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True)

text_output = scrolledtext.ScrolledText(
    output_frame, 
    width=100, 
    height=18, 
    font=('Segoe UI', 11), 
    wrap=tk.WORD, 
    bg="#FDF6EE", 
    fg="black", 
    insertbackground="black"
)
text_output.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

# Barre de statut améliorée
status_frame = ttk.Frame(main_frame, style='TFrame')
status_frame.pack(fill=tk.X, pady=(0, 5))

label_status = ttk.Label(
    status_frame, 
    text="Prêt", 
    style='Status.TLabel',
    foreground='white',
    anchor='center',
    justify='center',
    wraplength=650  # Permet le retour à la ligne automatique
)
label_status.pack(side=tk.LEFT)

# Pied de page
footer_frame = ttk.Frame(root, style='TFrame')
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

footer = tk.Label(
    footer_frame,
    text="Développé par Victor Carré — GitHub",
    font=("Segoe UI", 8, "italic"),
    fg="white",
    bg="#323232",
    cursor="hand1",
    anchor="w"
)
footer.pack(side=tk.LEFT, fill=tk.X, expand=True)
footer.bind("<Button-1>", open_github)

help_button = ttk.Button(
    footer_frame,
    text="?",
    command=show_help,
    width=2,
    style='Green.TButton'
)
help_button.pack(side=tk.RIGHT)

root.mainloop()