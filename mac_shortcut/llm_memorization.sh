#!/bin/bash

# Chemin absolu vers le dossier du projet
PROJECT_DIR="/Users/victorcarre/Code/Projects/llm-memorization"
CONFIG_FILE="$PROJECT_DIR/resources/config.json"

# Vérification existence de config.json
if [ ! -f "$CONFIG_FILE" ]; then
  osascript -e 'display alert "Erreur" message "Fichier config.json introuvable"'
  exit 1
fi

# Vérification outils de base
command -v jq >/dev/null 2>&1 || { osascript -e 'display alert "Erreur" message "jq non trouvé dans PATH"'; exit 1; }
command -v python3 >/dev/null 2>&1 || { osascript -e 'display alert "Erreur" message "python3 non trouvé dans PATH"'; exit 1; }

# Extraction des chemins depuis le JSON
LVMEMORY_FOLDER=$(jq -r '.lmstudio_folder_path' "$CONFIG_FILE")
SYNC_SCRIPT=$(jq -r '.sync_script_path' "$CONFIG_FILE")
PROJECT_SCRIPT=$(jq -r '.project_script_path' "$CONFIG_FILE")
DB_PATH=$(jq -r '.db_path' "$CONFIG_FILE")
VENV_ACTIVATE=$(jq -r '.venv_activate_path' "$CONFIG_FILE")

# Résolution des chemins
resolve_path() {
  local path="$1"
  [[ "$path" == ~* ]] && echo "$(eval echo "$path")" && return
  [[ "$path" == /* ]] && echo "$path" && return
  echo "$PROJECT_DIR/$path"
}

LVMEMORY_FOLDER_ABS=$(resolve_path "$LVMEMORY_FOLDER")
SYNC_SCRIPT_ABS=$(resolve_path "$SYNC_SCRIPT")
PROJECT_SCRIPT_ABS=$(resolve_path "$PROJECT_SCRIPT")
DB_PATH_ABS=$(resolve_path "$DB_PATH")
VENV_ACTIVATE_ABS=$(resolve_path "$VENV_ACTIVATE")

# Activation de l'environnement virtuel
if [ -f "$VENV_ACTIVATE_ABS" ]; then
  source "$VENV_ACTIVATE_ABS"
else
  osascript -e 'display alert "Erreur" message "Environnement virtuel non trouvé à '"$VENV_ACTIVATE_ABS"'"'
  exit 1
fi

# Exécution du script principal
if [ -f "$PROJECT_SCRIPT_ABS" ]; then
  python3 "$PROJECT_SCRIPT_ABS"
else
  osascript -e 'display alert "Erreur" message "Script Python introuvable à '"$PROJECT_SCRIPT_ABS"'"'
  exit 1
fi

# Notification finale si tout s’est bien passé
osascript -e 'display notification "Mémoire LLM exécutée avec succès" with title "Lancement terminé"'
