# LLM Memorization & Prompt Enhancer

> Un projet pour indexer, historiser et rechercher les conversations avec un LLM local (utilisé via une librairie comme LM Studio) avec une base SQLite enrichie par des mots-clés extraits automatiquement.

> Le projet est conçu pour fonctionner sans appel aux API externes, gardant la confidentialité des données.

> L'idée est de fournir un contexte important et personnalisé lors du début d'une conversation avec un LLM, en ajoutant des informations mémoires au premier prompt de l'échange.

______

## Objectifs

- Automatisation de requêtes depuis la librairie de gestion conversationnelle de LLMs locaux (LM Studio, Transformer Labs, Ollama, ...) pour constituer une base de donnée SQLite.
- Amélioration de prompts en proposant un contexte adapté à la question posée en s'appuyant sur les échanges précédents.
- Interface graphique tout-en-un.
______

## Fonctionnement

### 1. Extraction des conversations

Le script `import_lmstudio.py` explore le dossier de LM Studio, lit tous les `.json` de conversations et en extrait les paires `(question, réponse)`.

Chaque échange est :  
- Stocké dans la table `conversations`.  
- Hashé avec SHA-256 pour éviter les doublons.  
- Horodaté pour pouvoir retrouver des conversations en fonction du temps.  
- Analysé via **KeyBERT** pour en extraire 5 mots-clés, qui sont stockés dans la table `keywords`.

Utilisation de `import_lmstudio.py` :
- **Synchronisation** depuis l'outil `enhancer.py`.
- **Lancement manuel** avec `synchro_conversations.command`, qui rend le script **exécutable**.  
- **Automatiser avec `cron`** pour exécuter le script à intervalles réguliers.

### 2. Amélioration de prompts

Le script `enhancer.py` :

- Pose la question initiale,  
- Extrait les mots-clés correspondants,  
- Récupère les couples questions/réponses similaires dans la base SQL,  
- Résume les réponses avec un LLM local (sshleifer/distilbart-cnn-12-6),  
- Colle dans le presse-papiers un prompt complet contenant les précédents échanges résumés comme contexte, en terminant avec la question initiale.
- Offre une interface graphique avec pop-up d'aide,  
- Exécutable avec `prompt_enhancer.command`.

Remarque : le LLM local utilisé pour le raccourcissement du contexte peut être téléchargé en amont avec le script `model_download.py` (1.2Gb). 
Le choix du modèle `sshleifer/distilbart-cnn-12-6` été fait en prenant en compte sa taille, sa puissance, et ses besoins matériels (4 Go RAM libre nécessaire).
Le but était de trouver un équilibre pour éviter d'avoir des requêtes avec un temps d'attente supérieur à une dizaine de secondes. 
D'autres modèles moins lourds peuvent être utilisés au détriment de la rapidité, tel que `Falconsai/text_summarization` qui ne pèse que  ~250 Mb. Ce modèle peut être utilisé dans le  script `enhancer_light.py`, lancé avec `prompt_enhancer_light.command`.
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

5. Construction de l'arborescence

Aucun changement d'arborescence n'est nécessaire si les scripts sont exécutés depuis le repository cloné, hormis la ligne 10 de `import_lmstudio.py`.

Déplacer la base `conversations.db` depuis llm-memorization/datas/ vers le dossier désiré.
Adapter les chemins suivants :

- `SQL_memorization_base.ipynb`
 --> [Cellule 1] : db_path = '/chemin/relatif/vers/conversations.db'
- `import_lmstudio.py` 
--> ligne 9 : DB_PATH = '/chemin/relatif/vers/conversations.db'
--> ligne 10 : FOLDER_PATH = '~/user/.lmstudio/conversations'  # Chemin vers les fichiers .json conversationnels, souvent dans les fichiers cachés
- `enchancer.py`
--> ligne 17 : DB_PATH = '/chemin/relatif/vers/conversations.db'
--> ligne 29 : subprocess.run(["python3", "/chemin/relatif/vers/import_lmstudio.py"], check=True)

______

## Lancement
Adapter les chemins suivants :


- Pour lancer le script de synchronisation de mémoire et d'amélioration de prompts avec interface :
```bash
./prompt_enhancer_tkinter.command
```

______

## Remarque

- Ces scripts sont fonctionnels avec LM Studio, mais devraient pouvoir être adapté à tout software mettant à disposition les conversations au format `.json`.
