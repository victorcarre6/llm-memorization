import sqlite3
import re
import pyperclip
import tkinter as tk
import subprocess
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tkinter import scrolledtext, ttk
from transformers import pipeline, GPT2TokenizerFast, AutoModelForCausalLM
import torch
from keybert import KeyBERT
import os
import webbrowser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Constantes ---
TOP_K = 5

# --- Connexion à la base ---
DB_PATH = '/Users/victorcarre/Code/Projects/llm-memorization/datas/conversations.db' # Chemin vers la base de données SQLite, à adapter à votre path
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- Initialisation des modèles ---
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
kw_model = KeyBERT()

# Initialisation tokenizer sans SentencePiece (GPT2 tokenizer)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # important pour éviter les erreurs de padding

# Chargement modèle
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

summarizer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
# === FONCTIONS ===

def sync_conversations():
    try:
        subprocess.run(["python3", "/Users/victorcarre/Code/Projects/llm-memorization/scripts/import_lmstudio.py"], check=True)
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

def summarize_text(text, max_len=100, min_len=30):
    cleaned = clean_text(text)
    summary = summarizer(cleaned, max_length=max_len, min_length=min_len, do_sample=False)
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

def generate_prompt_paragraph(context, question, target_tokens=1000, tokenizer_name=None):
    # Nettoyage initial et vérification
    if not context or not isinstance(context, (list, tuple)):
        return f"Voici une nouvelle question à traiter : {question}"
    
    previous_questions = []
    keywords_used = set()
    conversation_summaries = []
    current_length = 0
    
    for item in context:
        try:
            user_input = str(item[0])[:200]
            llm_output = str(item[1])
            keyword = str(item[3]) if len(item) > 3 else None
            
            clean_keyword = None
            if keyword and keyword.lower() not in {"none", "null", "", "1", "2", "3"}:
                clean_keyword = keyword.lower().strip()
                keywords_used.add(clean_keyword)
            
            summary = clean_summary(summarize_text(llm_output))
            if summary:
                conversation_summaries.append({
                    'question': user_input,
                    'summary': summary,
                    'keyword': clean_keyword
                })
                
        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue
    
    parts = []
    
    if conversation_summaries:
        questions = [f"'{s['question']}'" for s in conversation_summaries]
        if len(questions) == 1:
            q_text = questions[0]
        elif len(questions) == 2:
            q_text = f"{questions[0]} ainsi que {questions[1]}"
        else:
            *init, last = questions
            q_text = ", ".join(init) + f", ainsi que {last}"
        
        intro = f"Au cours de tes discussions avec l'utilisateur, celui-ci t'avais déjà posé des questions sur {q_text}."
        parts.append(intro)
        current_length += len(intro.split())
    
    if keywords_used:
        keywords_str = ", ".join(sorted(k for k in keywords_used if k and len(k) > 2))
        if keywords_str:
            kw_part = f"Les thèmes abordés étaient : {keywords_str}."
            if current_length + len(kw_part.split()) <= target_tokens * 0.6:
                parts.append(kw_part)
                current_length += len(kw_part.split())
    
    if conversation_summaries:
        summary_text = "Les conversations ont permis d'établir que "
        summaries = []
        
        for s in conversation_summaries:
            text = ""
            if s['keyword']:
                text += f"concernant {s['keyword']}, "
            text += s['summary']
            summaries.append(text)
        
        if len(summaries) == 1:
            summary_text += summaries[0]
        else:
            *init, last = summaries
            summary_text += ", ".join(init) + f", et que {last}"
        
        summary_text = clean_french(summary_text)
        
        if current_length + len(summary_text.split()) <= target_tokens - 30:
            parts.append(summary_text)
    
    question_part = f"\n\nVoici maintenant une question à traiter proche de ces discussions : {question}"
    parts.append(question_part)
    
    final_prompt = " ".join(parts)
    final_prompt = final_prompt.replace("  ", " ").replace(" ,", ",").replace(" .", ".")
    
    return final_prompt


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

for style_name, config in style_config.items():
    style.configure(style_name, **config)

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