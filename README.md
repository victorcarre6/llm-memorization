# Local LLM Memorization

> A project to index, archive, and search conversations with a local LLM (accessed via a library like LM Studio) using a SQLite database enriched with automatically extracted keywords.

> The project is designed to work without calling any external APIs, ensuring data privacy.

> The idea is to provide substantial and personalized context at the start of a conversation with an LLM by adding memory information to the initial prompt.

> The collected data (database and generated prompts) can be analyzed directly within the script.

______

## Project

- Automate queries from local LLM conversational management libraries (LM Studio, Transformer Labs, Ollama, ...) to build a SQLite database.
- Hybrid search function to find the most relevant contexts in the database:
  -  Filter conversations using keywords extracted from the question,
  -  Use a vector index to measure semantic similarity.
- Enhance prompts by providing a tailored context based on the posed question relying on previous exchanges.
- All-in-one graphical user interface:
  - Adjustable number of extracted keywords and contexts via sliders.
  - Data visualization :
    - Information on the generated prompt based on keywords.
    - Insights on the conversation database.
- Supports both French and English conversations and prompts, for bilingual use.
    
______

## How it works

![image](https://github.com/user-attachments/assets/a3746d16-ebee-4807-8054-ccee6ef59f76)

### 1. Conversation extraction

The `sync_lmstudio.py` script scans the LM Studio conversations folder, reads all `.json` files, and extracts `(input, output, model)` datas.

Each exchange is:  
- Stored in the table `conversations`.  
- Hashed with MD5 to avoid duplicates.
- Timestamped to retrieve conversations by date/time. 
- Analysed using **KeyBERT** to extract 15 keywords, which are stored in the `keywords` table (in text and in vectors).

### 2. Prompt Enhancement

The `llm_cortexpander.py` script, executable via `llm_memorization.command` :

- Takes the initial question, 
- Extracts the corresponding keywords, 
- Retrieves similar question/answer pairs from the SQLite database,
  - Combining keyword filtering for fast targeting of relevant conversations and vector search for refinement,
- Summarizes the answers using a local model ([`plguillou/t5-base-fr-sum-cnndm`](https://huggingface.co/plguillou/t5-base-fr-sum-cnndm)),  
- Copies a complete prompt to the clipboard, containing summarized previous exchanges, ending with the original question,
- Provides a graphical interface including:
  - Help window,
  - Data analysis window.

![Capture d’écran 2025-06-14 à 15 36 25](https://github.com/user-attachments/assets/116c197c-9d8d-4654-ab16-cc0b601f46ce)
<img width="1604" alt="image" src="https://github.com/user-attachments/assets/e17b574b-bf9c-4275-977d-ea6b56406c7d" />

## Future Directions

- Enhance the existing database visualization to provide clearer insights into user data (e.g. interactive topic maps, conversation heatmaps).
- Transform the script into an LM Studio plugin, to enable seamless prompt enhancement and real-time analytics directly within the interface.
______

## Installation

1. Clone the repository

```bash
git clone https://github.com/victorcarre6/llm-memorization
cd llm-memorization
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

  - Then install the NLP models `fr_core_news_lg` and `en_core_web_lg`.

```bash
python -m spacy download fr_core_news_lg
python -m spacy download en_core_web_lg
```

4. Download the local model

  - Either with the dedicated script:

```bash
python scripts/model_download.py
```
  - Or via GitLFS (see `Notes` bellow)

5. Directory structure

- The `config.json` file at the root contains the paths required for the scripts to function properly.

______

## Launch

```bash
./llm_memorization.command
```

or read the README.md in `/mac_shortcut` to install a shortcut that launch the script directly from your dock.

______

## Notes

- These scripts work with LM Studio but can be adapted to any software providing conversations in `.json` format.

- The script applies a multiplier factor (default 2) to the requested number of extracted keywords to obtain more raw keywords, then filters irrelevant ones to ensure a sufficient, high-quality final set. This multiplier is configurable in `config.json` under `keyword_multiplier`.

### About the conversations.db database

- A French stop-word dictionary is used to eliminate irrelevant keywords (coordinating conjunctions, prepositions, etc.). The file `resources/stopwords_fr.json`can be modified to keep or remove specific keywords. This dictionary can be replaced with a custom file via the `stopwords_file_path` label in `config.json`.

- Example database: ~200 Q&A pairs with OpenChat-3.5, Mistral-7B and DeepSeek-Coder-6.7B on topics including :
  - Green Chemistry & Catalysis `(FR)`
  - Pharmaceutical Applications & AI `(FR)`
  - Photoactivable Molecules & Photocontrol `(FR)`
  - Plant Science & Biostimulants `(FR)`
  - Cross-disciplinary Tools `(FR)`
  - OLED Materials `(EN)`
  - Machine Learning in Agrochemistry `(EN)`
It is highly advised to build your own database in order to have prompts generated in a single language only.

- To avoid syncing conversations, they can be hidden in `~/.lmstudio/conversations/unsync`.

### Summarizing Model

The script uses the model [`plguillou/t5-base-fr-sum-cnndm`](https://huggingface.co/plguillou/t5-base-fr-sum-cnndm), selected for its good balance between performance and hardware requirements (4 GB of free RAM). This multilingual model allows summarization of both French and English conversations, keeping response times under 30 seconds.

You can configure the summarization model in config.json via the summarizing_model key, using either:

```bash
"summarizing_model": "plguillou/t5-base-fr-sum-cnndm"    // Hugging Face model (loaded online)
"summarizing_model": "resources/models/t5-base-fr-sum-cnndm"  // Local model (directory path)
```

If the Hugging Face model is unreachable (e.g. offline usage), the script will automatically fall back to the local model if the directory exists.

➤ Local model setup with GitLFS

1. Install Git LFS (if not already installed):

```bash
git lfs install
```
2. Pull the model from the repository:

```bash
git lfs pull
```

The model file will appear here: `resources/models/t5-base-fr-sum-cnndm/model.safetensors`

Make sure your config.json points to this local folder: `"summarizing_model": "resources/models/t5-base-fr-sum-cnndm"`
