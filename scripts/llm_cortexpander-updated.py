import heapq
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import platform
import warnings
import webbrowser
from collections import Counter
from dataclasses import dataclass
import tkinter as tk
from tkinter import scrolledtext, ttk
import faiss
import numpy as np
import pyperclip
import seaborn as sns
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer, logging as transformers_logging, pipeline
from keybert import KeyBERT
# === INITIALISATION ===

# --- Configuration projet & chargement du fichier config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(PROJECT_ROOT, "resources", "config.json")

def expand_path(value):
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.normpath(os.path.join(PROJECT_ROOT, expanded))
    return expanded

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    path_keys = {
        "venv_activate_path",
        "lmstudio_folder_path",
        "sync_script_path",
        "project_script_path",
        "db_path",
        "stopwords_file_path"
    }

    config = {}
    for key, value in raw_config.items():
        if isinstance(value, str):
            if key == "summarizing_model" and value == "model/barthez-orangesum-abstract":
                config[key] = expand_path(value)
            elif key in path_keys:
                config[key] = expand_path(value)
            else:
                config[key] = value
        else:
            config[key] = value
    return config

config = load_config(config_path)

# --- Logging & suppression des avertissements ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# --- Chargement des stopwords ---
stopwords_path = config["stopwords_file_path"]
with open(stopwords_path, "r", encoding="utf-8") as f:
    french_stop_words = set(json.load(f))

combined_stopwords = list(ENGLISH_STOP_WORDS.union(french_stop_words))

# --- Connexion à la base SQLite ---
db_path = config["db_path"]
if not os.path.exists(db_path):
    default_db_path = os.path.join(os.path.dirname(db_path), "conversations_example.db")
    print("Base de donnée d'example chargée.")
    if os.path.exists(default_db_path):
        config["db_path"] = default_db_path
    else:
        raise FileNotFoundError(f"Neither {db_path} nor {default_db_path} exist.")

conn = sqlite3.connect(config["db_path"])
cur = conn.cursor()

# --- Initialisation de l’index vectoriel ---
VECTOR_DIM = 384
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# --- Initialisation des modèles avec fallback modèle local ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = config.get("summarizing_model", "plguillou/t5-base-fr-sum-cnndm")
local_model_dir = os.path.join(PROJECT_ROOT, "resources", "models", "t5-base-fr-sum-cnndm")
try:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarizing_model = T5ForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    print("Modèle introuvable sur Hugging Face, chargement local en cours...")
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"Le modèle local {local_model_dir} est introuvable.")
    tokenizer = T5Tokenizer.from_pretrained(local_model_dir)
    summarizing_model = T5ForConditionalGeneration.from_pretrained(local_model_dir)
summarizing_pipeline = pipeline(task="summarization", model=summarizing_model, tokenizer=tokenizer, framework="pt")

# === STRUCTURES DE DONNÉES ===

@dataclass
class KeywordsData:
    kw_lemma: str
    weight: float
    freq: int
    score: float

@dataclass
class ContextData:
    user_input: str
    llm_output: str
    score_kw: float
    llm_model: str
    timestamp: str
    convo_id: str
    score_rerank: float

    @property
    def combined_score(self):
        return self.score_kw * self.score_rerank

final_prompt = ""

# === PIPELINE LINGUISTIQUE ===

class LanguagePipeline:
    def __init__(self, text):
        self.text = text
        self.lang = self.detect_language(text)
        self._nlp = None
        self._kw_model = None

    def detect_language(self, text):
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "fr"
        return lang

    @property
    def nlp(self):
        if self._nlp is None:
            if self.lang.startswith("en"):
                self._nlp = spacy.load("en_core_web_lg")
            else:
                self._nlp = spacy.load("fr_core_news_lg")
        return self._nlp

    @property
    def kw_model(self):
        if self._kw_model is None:
            if self.lang.startswith("en"):
                self._kw_model = KeyBERT(model=SentenceTransformer("allenai/scibert_scivocab_uncased"))
            else:
                self._kw_model = KeyBERT(model=SentenceTransformer("camembert-base"))
        return self._kw_model

    def lemmatize(self, word):
        return self.nlp(word)[0].lemma_.lower()

    def extract_keywords(self, keyphrase_ngram_range=(1, 1), stopwords=None, top_n=5):
        return self.kw_model.extract_keywords(
            self.text,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stopwords,
            top_n=top_n
        )

# === FONCTIONS TERTIAIRES : Mots-clés, Vecteurs, Formatage, NLP ===

# --- Paramètres et Interface ---
root = tk.Tk()
keyword_count_var = tk.IntVar(value=5)
context_count_var = tk.IntVar(value=3)
multiplier = config.get("keyword_multiplier", 2)
threshold = config.get("similarity_threshold", 0.2)

# --- Extraction de mots-clés ---
def extract_keywords(text, top_n=None):
    global filtered_keywords
    if top_n is None:
        top_n = keyword_count_var.get()

    pipeline = LanguagePipeline(text)

    # Extraction brute
    raw_keywords = pipeline.extract_keywords(
        keyphrase_ngram_range=(1, 1),
        stopwords=combined_stopwords,
        top_n=top_n * multiplier
    )

    # Lemmatisation + fréquence
    tokens = re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower())
    lemmatized_tokens = [pipeline.lemmatize(tok) for tok in tokens if tok not in combined_stopwords]
    token_freq = Counter(lemmatized_tokens)

    def is_valid_kw(kw):
        return (
            kw not in combined_stopwords and
            len(kw) > 2 and
            (kw.isalpha() or '-' in kw)
        )

    # Filtrage : unicité, lemmatisation, score pondéré
    filtered_raw, seen = [], set()
    for kw, weight in raw_keywords:
        kw_clean = kw.lower().strip()
        if is_valid_kw(kw_clean):
            kw_lemma = pipeline.lemmatize(kw_clean)
            if kw_lemma.endswith('s') and kw_lemma[:-1] in seen:
                continue
            if kw_lemma in seen:
                continue
            freq = token_freq.get(kw_lemma, 0)
            score = freq * weight
            filtered_raw.append((score, freq, kw_lemma, weight))
            seen.add(kw_lemma)

    top_filtered = heapq.nlargest(top_n, filtered_raw, key=lambda x: x[0])

    # Création objets KeywordsData
    filtered_keywords, seen = [], set()
    for score, freq, kw_lemma, weight in top_filtered:
        if kw_lemma not in seen:
            seen.add(kw_lemma)
            filtered_keywords.append(
                KeywordsData(
                    kw_lemma=kw_lemma,
                    weight=round(weight, 2),
                    freq=freq,
                    score=round(score, 2)
                )
            )

    return filtered_keywords

# --- Nettoyage de texte brut ---
def format_cleaner(contenu):
    lignes = contenu.split('\n')
    lignes_nettoyees = []
    for ligne in lignes:
        ligne = ligne.strip()
        ligne_modifiee = re.sub(r"^\s*[-*\d]+[\.\)]?\s*[^:]*:\s*", "", ligne)
        lignes_nettoyees.append(ligne_modifiee)
    return "\n".join(lignes_nettoyees)

# --- NLP : découpage et nettoyage pour résumé ---
def nlp_clean_text(text, max_chunk_size=500):
    text = re.sub(r'```(?:python)?\s*.*?```', '', text, flags=re.DOTALL)
    nlp = pipeline(text)
    chunks, current_chunk, current_length = [], [], 0

    for sent in nlp(text).sents:
        s = sent.text.strip()
        if len(s) < 20:
            continue
        if current_length + len(s) < max_chunk_size:
            current_chunk.append(s)
            current_length += len(s)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_length = len(s)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return " ".join(chunks[:3]), nlp  # max 3 blocs

# --- FONCTIONS SECONDAIRES : CONTEXTE ET PROMPT ---

def get_relevant_context(user_question, context_limit=None):
    global filtered_context

    # Détermination de la limite
    if context_limit is None:
        context_limit = context_count_var.get()

    # Étape 1 : extraction des mots-clés et encodage
    keywords = extract_keywords(user_question)
    keyword_strings = [
        kw.keyword if hasattr(kw, 'keyword') else (kw[0] if isinstance(kw, (list, tuple)) and kw else str(kw))
        for kw in keywords
    ]
    query_kw_vectors = np.array(embedding_model.encode(keyword_strings, convert_to_tensor=False)).astype('float32')

    # Étape 2 : extraction des vecteurs stockés depuis SQLite
    try:
        cur.execute("SELECT conversation_id, keyword, vector FROM vectors")
        vector_rows = cur.fetchall()
    except Exception:
        return []

    if not vector_rows:
        return []

    convo_ids, stored_kw_vectors = [], []
    for conv_id, keyword, vector in vector_rows:
        try:
            vec = np.frombuffer(vector, dtype='float32')
            if vec.shape == (384,):
                stored_kw_vectors.append(vec)
                convo_ids.append(conv_id)
        except Exception:
            continue

    if not stored_kw_vectors:
        return []

    stored_kw_vectors = np.vstack(stored_kw_vectors).astype('float32')

    # Vérifications
    if any(np.isnan(arr).any() or np.isinf(arr).any() for arr in [stored_kw_vectors, query_kw_vectors]):
        return []

    # Normalisation
    faiss.normalize_L2(stored_kw_vectors)
    faiss.normalize_L2(query_kw_vectors)

    query_kw_vectors /= np.linalg.norm(query_kw_vectors, axis=1, keepdims=True)
    stored_kw_vectors /= np.linalg.norm(stored_kw_vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(query_kw_vectors, stored_kw_vectors.T)

    # Recherche top-k
    k = min(context_limit, stored_kw_vectors.shape[0])
    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
    topk_scores = np.take_along_axis(similarity_matrix, topk_indices, axis=1)

    matched_convo_ids = {
        convo_ids[idx]
        for scores_row, indices_row in zip(topk_scores, topk_indices)
        for score, idx in zip(scores_row, indices_row) if score >= threshold
    }

    if not matched_convo_ids:
        return []

    # Calcul score par conversation
    convo_sim_scores = {cid: [] for cid in matched_convo_ids}
    for scores_row, indices_row in zip(topk_scores, topk_indices):
        for score, idx in zip(scores_row, indices_row):
            cid = convo_ids[idx]
            if cid in convo_sim_scores and score >= threshold:
                convo_sim_scores[cid].append(score)

    final_sim_scores = {cid: max(scores) for cid, scores in convo_sim_scores.items()}

    # Étape 3 : récupération des lignes de contexte
    placeholders_ids = ','.join(['?'] * len(matched_convo_ids))
    cur.execute(f'''
        SELECT user_input, llm_output, llm_model, timestamp, id
        FROM conversations
        WHERE id IN ({placeholders_ids})
    ''', list(matched_convo_ids))
    context_rows = cur.fetchall()

    if not context_rows:
        return []

    # Étape 4 : rerank par similarité directe avec la user_question
    user_inputs = [row[0] for row in context_rows]
    user_question_vec = embedding_model.encode([user_question], convert_to_tensor=False)
    user_question_vec = np.array(user_question_vec[0]).astype('float32')
    user_question_vec /= np.linalg.norm(user_question_vec)

    input_vectors = embedding_model.encode(user_inputs, convert_to_tensor=False)
    input_vectors = np.array(input_vectors).astype('float32')
    input_vectors /= np.linalg.norm(input_vectors, axis=1, keepdims=True)

    rerank_scores = np.dot(input_vectors, user_question_vec)

    # Construction des objets de contexte
    filtered_context = []
    for i, (user_input, llm_output, llm_model, timestamp, convo_id) in enumerate(context_rows):
        score_kw = final_sim_scores.get(convo_id, 0)
        score_rerank = rerank_scores[i]
        filtered_context.append(ContextData(
            user_input=user_input,
            llm_output=llm_output,
            score_kw=score_kw,
            llm_model=llm_model,
            timestamp=timestamp,
            convo_id=convo_id,
            score_rerank=score_rerank
        ))


    filtered_context.sort(key=lambda x: x.combined_score, reverse=True)
    return filtered_context


def summarize(text, focus_terms=None, max_length=512):
    update_status("⚙️ Shortening of extracted context ...")
    root.update()

    try:
        if focus_terms:
            sentences = [
                s.strip() for s in text.split('.') 
                if any(term.lower() in s.lower() for term in focus_terms)
            ]
            filtered_text = '. '.join(sentences)
            text = filtered_text if filtered_text else text
        text = text[:2000]

        result = summarizing_pipeline(
            text,
            max_new_tokens=max_length,
            min_length=min(50, max_length // 2),
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception as e:
        print(f"[ERREUR] summarization : {type(e).__name__} - {e}")
        return text[:max_length] + "... [résumé tronqué]"


def generate_prompt_paragraph(context, question, keywords=None, lang="fr"):
    global context_count
    update_status("⚙️ Prompt generation ...")
    root.update()

    if not context:
        context_count = 0
        return f"{question}"

    processed_items = []
    context_limit = context_count_var.get()
    context_count = 0

    for item in context[:context_limit]:
        try:
            user_input = str(item.user_input)[:300]
            llm_output = str(item.llm_output)
            summary = summarize(format_cleaner(llm_output))
            processed_items.append({
                'question': user_input,
                'summary': summary,
                'keyword': keywords if keywords else None
            })
        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue

    context_count = len(processed_items)

    if not processed_items:
        return question, context_count

    # Construction des blocs de prompt
    parts = []

    # Questions
    questions = [f"'{item['question']}'" for item in processed_items]
    if lang == "en":
        if len(questions) == 1:
            parts.append(f"Your previous conversations led you to answer the following question: {questions[0]}")
        else:
            *init, last = questions
            parts.append(f"Your previous conversations led you to answer the following questions: {', '.join(init)}, and finally {last}")
    else:
        if len(questions) == 1:
            parts.append(f"Tes discussions avec l'utilisateur t'ont amené à répondre à cette question : {questions[0]}")
        else:
            *init, last = questions
            parts.append(f"Tes discussions avec l'utilisateur t'ont amené à répondre à ces questions : {', '.join(init)}, et enfin {last}")



    # Résumés et question de base
    summaries = [f"- {item['summary']}" for item in processed_items]
    if lang == "en":
        parts.append("These discussions involved the following topics:\n" + "\n".join(summaries) + "\n")
        parts.append(f"Now, answer this question in the context of those previous discussions: {question}")
    else:
        parts.append("Ces intéractions vous ont amené à discuter de ces sujets :\n" + "\n".join(summaries) + "\n")
        parts.append(f"Réponds maintenant à cette question, dans le contexte de vos discussions précédentes : {question}")

    return "\n".join(parts)


# === FONCTIONS PRINCIPALES ===

# Synchronisation des conversations
def sync_conversations():
    sync_path = config.get("sync_script_path")

    update_status("⚙️ Processing...")
    root.update()

    if not sync_path:
        label_status.config(text="❌ Error while synchronizing (sync_script_path).", foreground='#ff6b6b')
        return False
    try:
        subprocess.run(["python3", sync_path], check=True)
        label_status.config(text="✅ Synchronization completed.", foreground='#599258')
        return True
    except subprocess.CalledProcessError:
        label_status.config(text="❌ Error while synchronizing (subprocess).", foreground='#ff6b6b')
        return False
    except FileNotFoundError:
        label_status.config(text="❌ Synchronization script not found.", foreground='#ff6b6b')
        return False

# Pércuteur
def on_ask():
    global final_prompt, context_count
    question = entry_question.get("1.0", "end-1c")
    if not question.strip():
        update_status("⚠️ Ask a question.", error=True)
        return
    
    update_status("⚙️ Processing...")
    root.update()
    
    try:
        start_time = time.time()
        #print(f"Language detected: {lang}")
        context = get_relevant_context(question, context_limit=context_count_var.get()) #", limit=context_count_var.get()" ajoutée slider contexte
        pipeline = LanguagePipeline(question)
        lang = pipeline.lang
        prompt = generate_prompt_paragraph(context, question, lang=lang)
        final_prompt = prompt
        
        pyperclip.copy(prompt)
        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, prompt)
        
        # Calcul des métriques
        token_count = len(prompt.split())
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        update_status(
        f"✅ Prompt generated ({token_count} tokens in {elapsed_time} sec) | Context used: {context_count} element{'s' if context_count_var.get() > 1 else ''}",
        success=True
)
    except Exception as e:
        update_status(f"❌ Erreur : {str(e)}", error=True)

def show_infos():
    global notebook, filtered_keywords, filtered_context
    info_window = tk.Toplevel(root)
    info_window.title("Information about the generated prompt and the database")
    info_window.geometry("800x800")
    container = tk.Frame(info_window, bg="#323232")
    container.pack(fill="both", expand=True)

    info_window.transient(root)
    info_window.grab_set()

    question = entry_question.get("1.0", "end-1c")
    if not question:
        update_status("⚠️ Saisissez une question avant de continuer.", error=True)
        info_window.destroy()
        return

    notebook = ttk.Notebook(container, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # Tab 1: Keywords Analysis
    single_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(single_tab, text="Keywords")

    ## Keywords from user_question

    if filtered_keywords:
        # Handle both KeywordsData objects and raw tuples
        if isinstance(filtered_keywords[0], KeywordsData):
            sorted_keywords = sorted(filtered_keywords, key=lambda kw: kw.score, reverse=True)
            kw_lemmas_sorted = [kw.kw_lemma for kw in sorted_keywords]
            freqs_sorted = [kw.freq for kw in sorted_keywords]
            weights_sorted = [kw.weight for kw in sorted_keywords]
            scores_sorted = [k.score for k in sorted_keywords]
        else:
            print("Error for retrieving datas from filtered_keywords")

        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
        bar_width = 0.25
        x = list(range(len(kw_lemmas_sorted)))

        # Décalages pour les 3 types de barres
        occurrence_x = [i - bar_width for i in x]
        weight_x = x
        score_x = [i + bar_width for i in x]

        # Tracer les 3 barres
        ax.bar(occurrence_x, freqs_sorted, width=bar_width, label="Frequency", color="#599258")
        ax.bar(weight_x, weights_sorted, width=bar_width, label="Weight", color="#a2d149")
        ax.bar(score_x, scores_sorted, width=bar_width, label="Score", color="#e08c26")

        # Mise en forme graphique
        ax.set_facecolor("#323232")
        fig.patch.set_facecolor("#323232")
        ax.set_xticks(x)
        ax.set_xticklabels(kw_lemmas_sorted, rotation=45, ha='right', color="white", fontsize=9)
        ax.tick_params(axis='y', colors="white", labelsize=10)
        ax.set_title("Frequency, weight, and score from initial prompt keywords", color="white", fontsize=9, fontweight='bold')
        ax.legend(facecolor="#323232", labelcolor="white")

        # Bordures blanches
        for spine in ax.spines.values():
            spine.set_color('white')

        fig.tight_layout()

        # Affichage dans l'interface Tkinter
        canvas = FigureCanvasTkAgg(fig, master=single_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
        plt.close(fig)

        ## Keywords from final_prompt

        prompt_keywords = extract_keywords(final_prompt, top_n=15)

        if prompt_keywords:
            if isinstance(prompt_keywords[0], KeywordsData):
                sorted_prompt_kw = sorted(prompt_keywords, key=lambda kw: kw.score, reverse=True)
                kw_prompt_lemmas = [kw.kw_lemma for kw in sorted_prompt_kw]
                freqs_prompt = [kw.freq for kw in sorted_prompt_kw]
                weights_prompt = [kw.weight for kw in sorted_prompt_kw]
                scores_prompt = [kw.score for kw in sorted_prompt_kw]
            else:
                print("Error for retrieving datas from prompt_keywords")

            fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=100)
            bar_width = 0.25
            x2 = list(range(len(kw_prompt_lemmas)))

            occurrence_x2 = [i - bar_width for i in x2]
            weight_x2 = x2
            score_x2 = [i + bar_width for i in x2]

            ax2.bar(occurrence_x2, freqs_prompt, width=bar_width, label="Frequency", color="#599258")
            ax2.bar(weight_x2, weights_prompt, width=bar_width, label="Weight", color="#a2d149")
            ax2.bar(score_x2, scores_prompt, width=bar_width, label="Score", color="#e08c26")

            ax2.set_facecolor("#323232")
            fig2.patch.set_facecolor("#323232")
            ax2.set_xticks(x2)
            ax2.set_xticklabels(kw_prompt_lemmas, rotation=45, ha='right', color="white", fontsize=9)
            ax2.tick_params(axis='y', colors="white", labelsize=10)
            ax2.set_title("Frequency, weight, and score from generated prompt keywords", color="white", fontsize=9, fontweight='bold')
            ax2.legend(facecolor="#323232", labelcolor="white")

            for spine in ax2.spines.values():
                spine.set_color('white')

            fig2.tight_layout()

            canvas2 = FigureCanvasTkAgg(fig2, master=single_tab)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(20, 0))
            plt.close(fig2)

        # --- Nouveau Tab : Heatmap Corrélation ---
    heatmap_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(heatmap_tab, text="Heatmap Correlation")

    if filtered_keywords and isinstance(filtered_keywords[0], KeywordsData):
        kw_lemmas = [kw.kw_lemma for kw in filtered_keywords]
        embeddings = embedding_model.encode(kw_lemmas)

        # Calcul des similarités cosinus et rescaling
        sim_matrix = cosine_similarity(embeddings)
        min_sim = sim_matrix.min()
        max_sim = sim_matrix.max()
        rescaled_sim = (sim_matrix - min_sim) / (max_sim - min_sim + 1e-8)  # éviter div/0

        # Création de la figure matplotlib
        fig_hm, ax_hm = plt.subplots(figsize=(10, 10), dpi=100)
        heatmap = sns.heatmap(rescaled_sim, 
                            xticklabels=kw_lemmas, 
                            yticklabels=kw_lemmas,
                            cmap="coolwarm", 
                            vmin=0.0, vmax=1.0,
                            annot=False, 
                            ax=ax_hm,
                            color="white",
                            cbar_kws={'label': 'Similarity', 'shrink': 0.65, 'aspect': 20},
                            square=True)

        # Configuration du style
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Similarity', color='white')  # Définir la couleur du label ici
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')
        ax_hm.set_title("Keywords semantic similarity", color="white", fontsize=10, fontweight='bold')
        ax_hm.tick_params(axis='x', colors="white", labelsize=8, pad=20)
        ax_hm.tick_params(axis='y', colors="white", labelsize=8)
        fig_hm.patch.set_facecolor("#323232")
        ax_hm.set_facecolor("#323232")
        plt.tight_layout(pad=3)
    
        # Configuration spécifique pour éviter les coupures
        fig_hm.subplots_adjust(
            left=0.25,  
            right=0.95,  
            bottom=0.25,
            top=1      
        )
    
        # Configuration de la colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

        # Cadre conteneur pour positionnement précis
        container = tk.Frame(heatmap_tab, bg="#323232")
        container.pack(fill='both', expand=True)
        
        # Frame pour positionnement en haut à droite
        top_right_frame = tk.Frame(container, bg="#323232")
        top_right_frame.pack(anchor='ne', expand=True, padx=20, pady=20)

        # Intégration matplotlib
        canvas_hm = FigureCanvasTkAgg(fig_hm, master=top_right_frame)
        canvas_hm.draw()
        canvas_widget = canvas_hm.get_tk_widget()
        canvas_widget.pack()
        
    else:
        tk.Label(heatmap_tab, 
                text="No valid data to display heatmap.", 
                fg="white", 
                bg="#323232").pack(pady=20)

    plt.close('all')

 # --- Onglet Base de Données ---

    stats_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(stats_tab, text="Database")

    frame_stats = tk.Frame(stats_tab, bg="#323232")
    frame_stats.pack(fill="both", expand=True, padx=20, pady=20)

    # --- Connexion à la base pour stats globales ---
    conn = sqlite3.connect(config["db_path"])
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM conversations")
    nb_conversations = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM vectors")
    nb_mots_clefs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT keyword) FROM vectors")
    nb_mots_clefs_uniques = cur.fetchone()[0]
    conn.close()

    # --- Affichage des statistiques globales ---
    tk.Label(frame_stats, text="Overall database statistics", fg="white", bg="#323232", font=("Segoe UI", 12, "bold")).pack(pady=(0, 20))
    tk.Label(frame_stats, text=f"Number of conversations : {nb_conversations}", fg="white", bg="#323232", font=("Segoe UI", 12)).pack(anchor="w", pady=2)
    tk.Label(frame_stats, text=f"Number of keywords : {nb_mots_clefs}", fg="white", bg="#323232", font=("Segoe UI", 12)).pack(anchor="w", pady=2)
    tk.Label(frame_stats, text=f"Number of unique keywords : {nb_mots_clefs_uniques}", fg="white", bg="#323232", font=("Segoe UI", 12)).pack(anchor="w", pady=2)

    # --- Requête pour mots-clés les plus fréquents ---
    conn = sqlite3.connect(config["db_path"])
    cur = conn.cursor()
    cur.execute("""
        SELECT keyword, COUNT(*) as freq
        FROM vectors
        GROUP BY keyword
        ORDER BY freq DESC
        LIMIT 20
    """)
    top_keywords = cur.fetchall()
    conn.close()

    # --- Cadre horizontal contenant tableau + graphique ---
    two_col_frame = tk.Frame(frame_stats, bg="#323232")
    two_col_frame.pack(expand=True, fill="both", pady=20)

    # --- Ligne des titres alignés horizontalement ---
    titles_frame = tk.Frame(two_col_frame, bg="#323232")
    titles_frame.pack(fill="x", padx=(0, 30), pady=(0, 5))

    header_font = ("Segoe UI", 10, "bold")
    
    # Titre colonne gauche : "Most frequent keywords"
    tk.Label(titles_frame, text="Most frequent keywords", font=header_font,
            fg="white", bg="#323232", anchor="w").pack(side="left", padx=(0,180))

    # Titre colonne droite : "Conversations by LLM models"
    tk.Label(titles_frame, text="Conversations by LLM models", font=header_font,
            fg="white", bg="#323232", anchor="w").pack(side="left", expand=True, pady=0)

    # --- Colonne gauche : tableau ---
    table_frame = tk.Frame(two_col_frame, bg="#323232")
    table_frame.pack(side="left", fill="y", padx=(0, 30))

    # --- En-têtes du tableau ---
    tk.Label(table_frame, text="Keywords", fg="white", bg="#323232",
            font=header_font, width=20, anchor="w").grid(row=0, column=0, sticky="w", padx=3, pady=2)
    tk.Label(table_frame, text="Frequency", fg="white", bg="#323232",
            font=header_font, width=10, anchor="w").grid(row=0, column=1, sticky="w", padx=3, pady=2)

    data_font = ("Segoe UI", 10)
    for i, (keyword, freq) in enumerate(top_keywords, start=1):
        tk.Label(table_frame, text=keyword, fg="white", bg="#323232",
                font=data_font, anchor="w", width=20).grid(row=i, column=0, sticky="w", padx=3, pady=1)
        tk.Label(table_frame, text=str(freq), fg="white", bg="#323232",
                font=data_font, anchor="w", width=10).grid(row=i, column=1, sticky="w", padx=3, pady=1)

    # --- Colonne droite : graphique matplotlib ---
    graph_frame = tk.Frame(two_col_frame, bg="#323232")
    graph_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))

    # --- Données pour graphique ---
    conn = sqlite3.connect(config["db_path"])
    cur = conn.cursor()
    cur.execute("SELECT llm_model, COUNT(*) FROM conversations GROUP BY llm_model")
    model_counts = cur.fetchall()
    conn.close()

    models = [row[0] for row in model_counts]
    counts = [row[1] for row in model_counts]


    fig_model, ax_model = plt.subplots(figsize=(5, 4.5), dpi=100, facecolor='none')
    bars = ax_model.bar(models, counts, color="#599258", width=0.5)
    ax_model.set_facecolor("#323232")
    for spine in ['bottom', 'left', 'top', 'right']:
        ax_model.spines[spine].set_visible(True)
        ax_model.spines[spine].set_color('white')
    ax_model.set_xticks([])
    ax_model.tick_params(axis='y', which='both', colors='white', labelsize=8)
    ax_model.set_ylabel("Count", color="white", fontsize=9)
    ax_model.set_xlabel("LLM Models", color="white", fontsize=9)
    plt.tight_layout()
    fig_model.patch.set_facecolor("#323232")


    # Affichage des valeurs sur les barres
    for bar, model in zip(bars, models):
        height = bar.get_height()
        # Nom du modèle (le long de la barre)
        ax_model.text(bar.get_x() + bar.get_width()/2, height/2,
                    model,
                    rotation=90,
                    ha='center', va='bottom',
                    color='white', fontsize=8)
        
        # Valeur au-dessus de la barre
        #ax_model.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.02,
                    #str(int(height)),
                    #ha='center', va='bottom',
                    #color='white', fontsize=9, weight='bold')
        
    # --- Intégration matplotlib dans tkinter ---
    canvas_model = FigureCanvasTkAgg(fig_model, master=graph_frame)
    canvas_model.draw()
    canvas_widget_model = canvas_model.get_tk_widget()
    canvas_widget_model.pack(expand=True, fill="both", pady=(0, 0))
    plt.close(fig_model)

# === INTERFACE TKINTER ===

def update_status(message, error=False, success=False):
    label_status.config(text=message)
    if error:
        label_status.config(foreground='#ff6b6b')
    elif success:
        label_status.config(foreground='#599258')
    else:
        label_status.config(foreground='white')

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")

def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.geometry("600x250")
    help_window.configure(bg="#323232")
    help_window.resizable(False, False)

    frame = tk.Frame(help_window, bg="#323232")
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        frame,
        text="LLM Memorization — Don't panic !",
        font=("Segoe UI", 12, "bold"),
        bg="#323232",
        fg="white",
        justify=tk.CENTER
    )
    title_label.pack(fill=tk.X, pady=(0, 2))

    help_text = (
        "• Sync conversations: Adds the latest exchanges from LM Studio to your local database.\n\n"
        "• Generate prompt: Extracts the keywords from your question, searches for similar past conversations in your SQL database, and summarizes the relevant content using a local LLM.\n"
        "   The final prompt is displayed and automatically copied to your clipboard!\n\n"
        "• More: Opens an advanced statistics panel, including:\n"
        "   Visualization of keywords extracted from your question and from the generated prompt.\n"
        "   Correlation graphs between the keywords.\n"
        "   Database insights: number and frequency of keywords, and used LLM models.\n\n"
        "To learn more, troubleshoot potential script issues, or get in touch, visit:\n"
        "github.com/victorcarre6/llm-memorization."
    )
    label = tk.Label(
        frame,
        text=help_text,
        font=("Segoe UI", 12),
        bg="#323232",
        fg="white",
        justify=tk.LEFT,
        wraplength=550
    )
    label.pack(fill=tk.BOTH, expand=True)


def bring_to_front():
    root.update()
    root.deiconify()            
    root.lift()               
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False)) 


# === CONFIGURATION DE L'INTERFACE ===
root.title("LLM Memorization")
root.geometry("800x300")
root.configure(bg="#323232")

# Style global unique
style = ttk.Style(root)
style.theme_use('clam')

# Configuration du style
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 13),
        'padding': 2
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 11),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 10, 'italic underline'),
        'padding': 0
    },
    'TLabel': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 13)
    },
    'TEntry': {
        'fieldbackground': '#FDF6EE',
        'foreground': 'black',
        'font': ('Segoe UI', 13)
    },
    'TFrame': {
        'background': '#323232'
    },
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 13)
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
        'font': ('Segoe UI', 12),
        'bordercolor': '#323232',
        'borderwidth': 0,
    },
    'Custom.Treeview.Heading': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 13, 'bold'),
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

style.map('Bottom.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])


# === WIDGETS PRINCIPAUX ===
main_frame = ttk.Frame(root, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Section question - Centrée
question_header = ttk.Frame(main_frame, style='TFrame')
question_header.pack(fill='x', pady=(0, 1))
ttk.Label(question_header, text="Ask a question :").pack(expand=True)

question_frame = tk.Frame(main_frame, bg="#323232")
question_frame.pack(pady=(0, 5), fill='x', expand=True)

entry_question = tk.Text(question_frame, height=4, width=80, wrap="word", font=('Segoe UI', 13))
entry_question.pack(side="left", fill="both", expand=True)

# Scrollbar personnalisée
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
    style="Vertical.TScrollbar"
)
scrollbar.pack(side="right", fill="y")
entry_question.config(yscrollcommand=scrollbar.set)

entry_question.bind("<Return>", lambda event: on_ask())


# === CONTROLS & SLIDERS ===

control_frame = ttk.Frame(main_frame, style='TFrame')
control_frame.pack(fill='x', pady=(0, 10), padx=5)

slider_keywords_frame = ttk.Frame(control_frame, style='TFrame')
slider_keywords_frame.grid(row=0, column=0, sticky='w')

label_keyword_count = ttk.Label(slider_keywords_frame, text=f"Number of keywords: {keyword_count_var.get()}", style='TLabel')
label_keyword_count.pack(anchor='w')

slider_keywords = ttk.Scale(
    slider_keywords_frame,
    from_=1,
    to=15,
    orient="horizontal",
    variable=keyword_count_var,
    length=140,
    command=lambda val: label_keyword_count.config(text=f"Number of keywords: {int(float(val))}")
)
slider_keywords.pack(anchor='w')

slider_context_frame = ttk.Frame(control_frame, style='TFrame')
slider_context_frame.grid(row=0, column=1, padx=20, sticky='w')

label_contexts_count = ttk.Label(slider_context_frame, text=f"Number of contexts: {context_count_var.get()}", style='TLabel')
label_contexts_count.pack(anchor='w')

slider_contexts = ttk.Scale(
    slider_context_frame,
    from_=1,
    to=10,
    orient=tk.HORIZONTAL,
    variable=context_count_var,
    length=140,
    command=lambda val: label_contexts_count.config(text=f"Number of contexts: {int(float(val))}")
)
slider_contexts.pack(anchor='w')

button_frame = ttk.Frame(control_frame, style='TFrame')
button_frame.grid(row=0, column=2, sticky='e')

sync_button = ttk.Button(button_frame, text="Synchronize conversations",
                         command=sync_conversations, style='Green.TButton')
sync_button.pack(side='left', padx=5)

btn_ask = ttk.Button(button_frame, text="Generate prompt", command=on_ask, style='Green.TButton')
btn_ask.pack(side='left', padx=5)
control_frame.grid_columnconfigure(2, weight=1)


# === ZONE DE SORTIE ÉTENDABLE ===
output_expanded = tk.BooleanVar(value=False)

def toggle_output():
    if output_expanded.get():
        text_output.pack_forget()
        toggle_btn.config(text="▼ Show result")
        output_expanded.set(False)
        root.geometry("800x305")
    else:
        text_output.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        toggle_btn.config(text="▲ Hide result")
        output_expanded.set(True)
        root.geometry("800x750")

output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

toggle_btn = ttk.Button(
    output_frame,
    text="▼ Show result",
    command=toggle_output,
    style='Green.TButton'
)
toggle_btn.pack(fill=tk.X, pady=(0, 5))

text_output = scrolledtext.ScrolledText(
    output_frame,
    width=100,
    height=20,
    font=('Segoe UI', 13),
    wrap=tk.WORD,
    bg="#FDF6EE",
    fg="black",
    insertbackground="black"
)

# === MENU CONTEXTE (clic droit) ===

# Détection de l'OS
if platform.system() == "Darwin":
    right_click_event = "<Button-2>"
else:
    right_click_event = "<Button-3>"

output_context_menu = tk.Menu(text_output, tearoff=0)
output_context_menu.add_command(label="Copier", command=lambda: text_output.event_generate("<<Copy>>"))
output_context_menu.add_command(label="Coller", command=lambda: text_output.event_generate("<<Paste>>"))
output_context_menu.add_command(label="Tout sélectionner", command=lambda: text_output.tag_add("sel", "1.0", "end"))

def show_output_context_menu(event):
    try:
        output_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        output_context_menu.grab_release()

text_output.bind(right_click_event, show_output_context_menu)

# Menu contextuel pour entry_question (zone de question)
question_context_menu = tk.Menu(entry_question, tearoff=0)
question_context_menu.add_command(label="Copier", command=lambda: entry_question.event_generate("<<Copy>>"))
question_context_menu.add_command(label="Coller", command=lambda: entry_question.event_generate("<<Paste>>"))
question_context_menu.add_command(label="Tout sélectionner", command=lambda: entry_question.tag_add("sel", "1.0", "end"))

def show_question_context_menu(event):
    try:
        question_context_menu.tk_popup(event.x_root, event.y_root)
    finally:
        question_context_menu.grab_release()

entry_question.bind(right_click_event, show_question_context_menu)

# Ajouter aussi aux frames si nécessaire
question_frame.bind(right_click_event, show_question_context_menu)
output_frame.bind(right_click_event, show_output_context_menu)


# === BARRE DE STATUT ET BOUTONS ===
status_buttons_frame = ttk.Frame(main_frame, style='TFrame')
status_buttons_frame.pack(fill=tk.X, pady=(5, 2))

label_status = ttk.Label(
    status_buttons_frame,
    text="Ready",
    style='Status.TLabel',
    foreground='white',
    anchor='w'
)
label_status.pack(side=tk.LEFT, anchor='w')

right_buttons = ttk.Frame(status_buttons_frame, style='TFrame')
right_buttons.pack(side=tk.RIGHT, anchor='e')

btn_info = ttk.Button(right_buttons, text="More", style='Bottom.TButton', command=show_infos, width=8)
btn_info.pack(side=tk.TOP, pady=(0, 3))

btn_help = ttk.Button(right_buttons, text="Help", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack(side=tk.TOP)


# === FOOTER ===
footer_frame = ttk.Frame(root, style='TFrame')
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

dev_label = ttk.Label(footer_frame, text="Developped by Victor Carré —", style='TLabel', font=('Segoe UI', 10))
dev_label.pack(side=tk.LEFT)

github_link = ttk.Label(footer_frame, text="GitHub", style='Blue.TLabel', cursor="hand2")
github_link.pack(side=tk.LEFT)
github_link.bind("<Button-1>", open_github)

bring_to_front()

root.mainloop()