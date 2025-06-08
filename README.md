# LLM Memorization & Prompt Enhancer

> Un projet pour indexer, historiser et rechercher les conversations avec un LLM local (utilisé via une librairie comme LM Studio) avec une base SQLite enrichie par des mots-clés extraits automatiquement.

> Le projet est conçu pour fonctionner sans appel aux API externes, gardant la confidentialité des données.

> L'idée est de fournir un contexte important et personnalisé lors du début d'une conversation avec un LLM, en ajoutant des informations passées lors du premier prompt de l'échange.

______

## Objectifs

Automatisation de requêtes depuis la librairie de gestion de LLMs (LM Studio, Transformer Labs, Ollama, ...) afin de se servir de la base de donnée comme d'une mémoire à long terme.

- Mémoriser automatiquement les conversations avec un LLM local (ex. : LM Studio).  
- Éviter les doublons grâce à un système de hachage SHA-256.  
- Extraire et stocker des mots-clés pertinents avec **KeyBERT** pour la recherche.  
- Faciliter l’exploration des échanges passés via SQL ou un front-end à venir.
- Ajouter du contexte à des prompts désirés.

______

## Fonctionnement

### 1. Extraction des conversations

Le script `import_lmstudio.py` explore le dossier de LM Studio, lit tous les `.json` de conversations et en extrait les paires `(question, réponse)`.

Chaque échange est :  
- Stocké dans la table `conversations`.  
- Hashé avec SHA-256 pour éviter les doublons.  
- Horodaté pour pouvoir retrouver des conversations en fonction du temps.  
- Analysé via **KeyBERT** pour en extraire 5 mots-clés (n-grammes 1 à 2).  
- Les mots-clés sont stockés dans la table `keywords`.

### 2. Utilisation

a. Deux options d'extractions avec `import_lmstudio.py` :  
- **Lancement manuel** avec `synchro_conversations.command`, qui rend le script **exécutable**.  
- **Automatiser avec `cron`** pour exécuter le script à intervalles réguliers.

b. Utilisation de l'outil `enhancer.py` : contient la fonction de synchronisation de la base de donnée via une requête vers `import_lmstudio.py`

### 3. Amélioration de prompts

Le script `enhancer.py` :
- Interface graphique d'amélioration de prompts,  
- Pose la question initiale,  
- Extrait les mots-clés correspondants,  
- Récupère les couples questions/réponses similaires dans la base SQL,  
- Résume les réponses avec un LLM local (sshleifer/distilbart-cnn-12-6),  
- Colle dans le presse-papiers un prompt complet contenant les précédents échanges résumés comme contexte, avec la question initiale à la fin.
- Exécutable avec `prompt_enhancer.command`.

Remarque : le LLM local utilisé pour le raccourcissement du contexte peut être téléchargé (1.2Gb) en amont avec le script `model_download.py`.

______

## Installation

1. **Cloner le repo :**

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

4. Télécharger le modèle local

```bash
python scripts/model_download.py
```

______

## Lancement

- Pour lancer le script de synchronisation de mémoire et d'amélioration de prompts avec interface :
```bash
./prompt_enhancer_tkinter.command
```

______

## Remarques

- Le dossier par défaut de LM Studio est à configurer dans import_lmstudio.py.
- Le script enhancer_llm.py copie le prompt enrichi dans le presse-papiers, prêt à être collé dans LM Studio ou autre.
