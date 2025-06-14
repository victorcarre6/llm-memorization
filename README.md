# LLM Memorization

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
    
______

## How it works

![image](https://github.com/user-attachments/assets/a3746d16-ebee-4807-8054-ccee6ef59f76)

### 1. Conversation extraction

The `import_lmstudio.py` script scans the LM Studio conversations folder, reads all `.json` files, and extracts `(input, output, model)` datas.

Each exchange is:  
- Stored in the table `conversations`.  
- Hashed with MD5 to avoid duplicates.
- Timestamped to retrieve conversations by date/time. 
- Analysed using **KeyBERT** to extract 15 keywords, which are stored in the `keywords` table (in text and in vectors).

### 2. Prompt Enhancement

The `enhancer.py` script, executable via `prompt_enhancer.command` :

- Takes the initial question, 
- Extracts the corresponding keywords, 
- Retrieves similar question/answer pairs from the SQLite database,
  - Combining keyword filtering for fast targeting of relevant conversations and vector search for refinement,
- Summarizes the answers using a local model ([`plguillou/t5-base-fr-sum-cnndm`](https://huggingface.co/plguillou/t5-base-fr-sum-cnndm)),  
- Copies a complete prompt to the clipboard, containing summarized previous exchanges, ending with the original question,
- Provides a graphical interface including:
  - Help window,
  - Data analysis window.

 ![Capture d’écran 2025-06-13 à 20 51 50](https://github.com/user-attachments/assets/0ec61f92-eda2-40c5-8579-99bee84e6204)
<img width="1604" alt="image" src="https://github.com/user-attachments/assets/e17b574b-bf9c-4275-977d-ea6b56406c7d" />


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
  - Or via GitLFS (in the `model` folder)

5. Directory structure

The `config.json` file at the root contains the paths required for the scripts to function properly.

______

## Launch

```bash
./llm_memorization.command
```
______

## Notes

- These scripts work with LM Studio but can be adapted to any software providing conversations in `.json` format.

- The choice of [`plguillou/t5-base-fr-sum-cnndm`](https://huggingface.co/plguillou/t5-base-fr-sum-cnndm) was based on its size, power, and hardware requirements (4 GB free RAM needed). The main goal was to find a good compromise to keep query response times under about ten seconds. This model is multilingual, allowing the script to work with both French and English conversations. 

- he model can be changed by inserting a Hugging Face link in  `config.json` under the `model` label.

- LThe script applies a multiplier factor (default 2) to the requested number of extracted keywords to obtain more raw keywords, then filters irrelevant ones to ensure a sufficient, high-quality final set. This multiplier is configurable in `config.json` under `keyword_multiplier`.

- A French stop-word dictionary is used to eliminate irrelevant keywords (coordinating conjunctions, prepositions, etc.). The file `data/stopwords_fr.json`can be modified to keep or remove specific keywords. This dictionary can be replaced with a custom file via the `stopwords_file_path` label in `config.json`.

- Example database: ~200 Q&A pairs with OpenChat-3.5, Mistral-7B and DeepSeek-Coder-6.7B on topics including :
  - Green Chemistry & Catalysis `(FR)`
  - Pharmaceutical Applications & AI `(FR)`
  - Photoactivable Molecules & Photocontrol `(FR)`
  - Plant Science & Biostimulants `(FR)`
  - Cross-disciplinary Tools `(FR)`
  - OLED Materials `(EN)`
  - Machine Learning in Agrochemistry `(EN)`

- To avoid syncing conversations, they can be hidden in `~/.lmstudio/conversations/unsync`.
