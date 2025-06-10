# LLM Memorization & Prompt Enhancer

> Un projet pour indexer, historiser et rechercher les conversations avec un LLM local (utilisé via une librairie comme LM Studio) avec une base SQLite enrichie par des mots-clés extraits automatiquement.

> Le projet est conçu pour fonctionner sans appel aux API externes, gardant la confidentialité des données.

> L'idée est de fournir un contexte important et personnalisé lors du début d'une conversation avec un LLM, en ajoutant des informations mémoires au premier prompt de l'échange.

______

## Objectifs

- Automatisation de requêtes depuis la librairie de gestion conversationnelle de LLMs locaux (LM Studio, Transformer Labs, Ollama, ...) pour constituer une base de donnée SQLite.
- Amélioration de prompts en proposant un contexte adapté à la question posée en s'appuyant sur les échanges précédents.
- Interface graphique tout-en-un.
  - Choix du nombre de mots-clefs et de contextes extraits avec des sliders.
- Visualisation de données
  - Informations sur le prompt généré en fonction des mots clefs.
  - Informations sur la base de données de conversations (nuages de mots clefs, cartes mentales).
    
______

## Fonctionnement

![image](https://github.com/user-attachments/assets/a1ac907e-f830-4b99-934a-50b2394a248b)

### 1. Extraction des conversations

Le script `import_lmstudio.py` explore le dossier de conversations de LM Studio, lit tous les `.json` et en extrait les paires `(question, réponse)`.

Chaque échange est :  
- Stocké dans la table `conversations`.  
- Hashé avec MD5 pour éviter les doublons.  
- Horodaté pour pouvoir retrouver des conversations en fonction du temps.  
- Analysé via **KeyBERT** pour en extraire 20 mots-clés, qui sont stockés dans la table `keywords`.

### 2. Amélioration de prompts

Le script `enhancer.py`, exécutable avec `prompt_enhancer.command` :

- Pose la question initiale,  
- Extrait les mots-clés correspondants,  
- Récupère les couples questions/réponses similaires dans la base SQLite,  
- Résume les réponses avec un modèle local ([`moussaKam/barthez-orangesum-abstract`](https://huggingface.co/moussaKam/barthez-orangesum-abstract)),  
- Colle dans le presse-papiers un prompt complet, contenant les précédents échanges résumés, en terminant avec la question initiale,
- Offre une interface graphique avec :
  - fenêtre d'aide,
  - fenêtre d'analyse de données.
______

## Installation

1. Cloner le repository

```bash
git clone https://github.com/victorcarre6/llm-memorization
cd llm-memorization
```

2. Créer un venv

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

  - puis installer le modèle NLP `fr_core_news_lg`:

```bash
python -m spacy download fr_core_news_lg
```

4. Télécharger le modèle local

  - avec le script dédié : 

```bash
python scripts/model_download.py
```
  - ou avec GitLFS (dans le dossier `model`)

5. Arborescence

Le fichier `config.json` à la racine du repo contient les chemins nécessaires au bon fonctionnement des scripts. 

______

## Lancement

```bash
./llm_memorization.command
```
______

## Remarque

- Ces scripts sont fonctionnels avec LM Studio, mais devraient pouvoir être adapté à tout software mettant à disposition les conversations au format `.json`.

- Le modèle utilisé pour le raccourcissement du contexte est situé dans `/model`.

- Le choix du modèle [`moussaKam/barthez-orangesum-abstract`](https://huggingface.co/moussaKam/barthez-orangesum-abstract) été fait en prenant en compte sa taille, sa puissance, et ses besoins matériels (4 Go RAM libre nécessaire). L’objectif principal était de trouver un bon compromis afin d’éviter que les requêtes aient un temps d’attente supérieur à une dizaine de secondes. Ce modèle est multilingue, ce qui permet au script de fonctionner aussi bien avec des conversations en français qu’en anglais. 

- Il est possible de changer le modèle utilisé en insérant un lien Hugging Face dans le fichier le `config.json`  sous le label `model`.

- Le script applique un facteur multiplicateur (par défaut 2) au nombre de keyword extraits demandé, afin d’extraire plus de mots-clés bruts. Cela permet ensuite de filtrer et supprimer les mots-clés non pertinents, garantissant ainsi un nombre final suffisant et de qualité. Ce coefficiant est modifiable dans `config.json` sous le label `keyword_multiplier`.

- Un dictionnaire de « stop-words » français est utilisé pour éliminer les mots-clés non pertinents (onjonctions de coordinations, prépositions, etc.).
Le fichier `data/stopwords_fr.json` est modifiable si certains mots-clés présents dans la liste doivent être conservés ou retirés.
Ce dictionnaire peut être remplacé par un fichier personnalisé, dans `config.json` sous le label `stopwords_file_path`.
