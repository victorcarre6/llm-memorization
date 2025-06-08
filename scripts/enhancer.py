import sqlite3
import re
import pyperclip
import tkinter as tk
import subprocess
from tkinter import scrolledtext, ttk
from transformers import pipeline
from keybert import KeyBERT
import os
import webbrowser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Constantes ---
TOP_K = 5

# --- Connexion à la base ---
DB_PATH = '/Users/victorcarre/Code/Projects/llm-memorization/datas/conversations.db'
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- Initialisation des modèles ---
kw_model = KeyBERT()  # modèle d'extraction de mots-clés local
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# === FONCTIONS ===

def sync_conversations():
    try:
        subprocess.run(["python3", "/Users/victorcarre/Code/Projects/llm-memorization/scripts/import_lmstudio.py"], check=True)
        label_status.config(text="Synchronisation terminée.")
    except subprocess.CalledProcessError:
        label_status.config(text="Erreur lors de la synchronisation.")

def extract_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, _ in keywords]

def get_relevant_context(user_question, limit=TOP_K):
    keywords = extract_keywords(user_question)
    print(f"Mots-clés extraits de la question : {keywords}")
    if not keywords:
        return []
    placeholders = ', '.join(['?'] * len(keywords))
    query = f'''
        SELECT c.user_input, c.llm_output, c.timestamp
        FROM conversations c
        JOIN keywords k ON c.id = k.conversation_id
        WHERE k.keyword IN ({placeholders})
        ORDER BY c.timestamp DESC
        LIMIT ?
    '''
    cur.execute(query, (*keywords, limit))
    return cur.fetchall()

def clean_text(text):
    # Supprime les blocs de code délimités par ```python``` et ```
    cleaned = re.sub(r'```python.*?```', '', text, flags=re.DOTALL)
    return cleaned.strip()

def summarize_text(text, max_len=100, min_len=30):
    cleaned = clean_text(text)
    summary = summarizer(cleaned, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def generate_prompt(context, question):
    prompt = "Voici le contexte pertinent extrait des conversations passées :\n\n"
    for i, (user_input, llm_output, timestamp) in enumerate(context, 1):
        summary = summarize_text(llm_output)  # appel OK ici
        prompt += f"[Contexte {i} - {timestamp}]\nQuestion: {user_input}\nRéponse résumée: {summary}\n\n"
    prompt += f"Question à traiter : {question}\n\nRéponds de façon claire et synthétique."
    return prompt

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")

def show_help():
    help_text = (
        "LLM Memorization and Prompt Enhancer — Aide\n\n"
        "• Synchroniser les conversations : ajoute les nouveaux échanges depuis LM Studio à la base de données.\n\n"
        "• Générer prompt : extrait les mots-clés de votre question, cherche des conversations similaires dans votre base SQL, puis compresse les informations avec un LLM local.\n\n"
        "Le prompt final est affiché puis automatiquement copié dans votre presse-papier !\n\n"
        "Pour en savoir plus ou obtenir plus d'informations à propos d'un éventuel bloquage des scripts, visitez : github.com/victorcarre6/llm-memorization"
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
        label_status.config(text="⚠️ Merci de saisir une question.")
        return
    label_status.config(text="⌛ Traitement en cours...")
    root.update()

    context = get_relevant_context(question)
    prompt = generate_prompt(context, question)
    pyperclip.copy(prompt)

    text_output.delete('1.0', tk.END)
    text_output.insert(tk.END, prompt)
    label_status.config(text=" Prompt généré et copié dans le presse-papiers.")

root = tk.Tk()
root.title("LLM Memorization and Prompt Enhancer")
root.geometry("850x600")
root.configure(bg="#323232")

style = ttk.Style(root)
style.theme_use('clam')
style.configure('Green.TButton',
                background='#599258',
                foreground='white',
                font=('Segoe UI', 11),
                padding=6)
style.configure('TLabel',
                background='#323232',
                foreground='white',
                font=('Segoe UI', 11))

style.configure('TEntry',
                fieldbackground='#FDF6EE',
                foreground='black',
                font=('Segoe UI', 11))

style.configure('TFrame', background='#323232')

style.configure('Status.TLabel',
                background='#323232',
                font=('Segoe UI', 11))

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

# Question
ttk.Label(root, text="Poser la question :").pack(padx=10, pady=(10, 0))
entry_question = ttk.Entry(root, width=80, style='TEntry')
entry_question.pack(padx=10, pady=(0, 10))
entry_question.bind("<Return>", lambda event: on_ask())


# Boutons

button_frame = ttk.Frame(root, style='TFrame')
button_frame.pack(pady=10)

sync_button = ttk.Button(button_frame, text="Synchroniser les conversations", command=sync_conversations, style='Green.TButton')
sync_button.pack(side=tk.LEFT, padx=10)

btn_ask = ttk.Button(button_frame, text="Générer prompt", command=on_ask, style='Green.TButton')
btn_ask.pack(side=tk.LEFT, padx=10)

bottom_right_frame = ttk.Frame(root, style='TFrame')
bottom_right_frame.pack(side=tk.BOTTOM, anchor="se", padx=10, pady=10)

# Zone texte pour affichage
text_output = scrolledtext.ScrolledText(root, width=100, height=18, font=('Segoe UI', 11), wrap=tk.WORD, bg="#FDF6EE", fg="black", insertbackground="black")
text_output.pack(padx=20, pady=(10, 5))

# Label status
label_status = ttk.Label(root, text="", style='Status.TLabel')
label_status.pack(pady=(5, 5))

# Frame du bas complète (droite + gauche)
bottom_frame = tk.Frame(root, bg="#323232")
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

# Label de crédit aligné à gauche
footer = tk.Label(bottom_frame,
                  text="Développé par Victor Carré — GitHub",
                  font=("Segoe UI", 8, "italic"),
                  fg="white",
                  bg="#323232",
                  cursor="hand1",
                  anchor="w",  # alignement à gauche
                  justify="left")
footer.pack(side=tk.LEFT, fill=tk.X, expand=True)
footer.bind("<Button-1>", open_github)

# Bouton aide aligné à droite
help_button = ttk.Button(
    bottom_frame,
    text="?",
    command=show_help,
    width=2,
    style='Green.TButton'
)
help_button.pack(side=tk.RIGHT)

root.mainloop()
