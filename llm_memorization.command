#!/bin/bash

# Chemin vers la racine du projet (dossier où est ce script)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Charger la config.json dans des variables shell
CONFIG_FILE="$PROJECT_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Erreur : fichier config.json introuvable dans $PROJECT_DIR"
  read -p "Press Enter to close..."
  exit 1
fi

# Utiliser jq pour extraire les chemins
LVMEMORY_FOLDER=$(jq -r '.lmstudio_folder_path' "$CONFIG_FILE")
SYNC_SCRIPT=$(jq -r '.sync_script_path' "$CONFIG_FILE")
PROJECT_SCRIPT=$(jq -r '.project_script_path' "$CONFIG_FILE")
DB_PATH=$(jq -r '.db_path' "$CONFIG_FILE")
VENV_ACTIVATE=$(jq -r '.venv_activate_path' "$CONFIG_FILE")

# Fonction pour gérer les chemins absolus ou relatifs, avec expansion ~
resolve_path() {
  local path="$1"
  if [[ "$path" == ~* ]]; then
    # Expand home directory if path starts with ~
    echo "$(eval echo "$path")"
  else
    # Sinon concatène avec PROJECT_DIR
    echo "$PROJECT_DIR/$path"
  fi
}

# Appliquer la fonction pour chaque chemin
LVMEMORY_FOLDER_ABS=$(resolve_path "$LVMEMORY_FOLDER")
SYNC_SCRIPT_ABS=$(resolve_path "$SYNC_SCRIPT")
PROJECT_SCRIPT_ABS=$(resolve_path "$PROJECT_SCRIPT")
DB_PATH_ABS=$(resolve_path "$DB_PATH")
VENV_ACTIVATE_ABS=$(resolve_path "$VENV_ACTIVATE")

# Activer l'environnement virtuel
if [ -f "$VENV_ACTIVATE_ABS" ]; then
  source "$VENV_ACTIVATE_ABS"
else
  echo "Erreur : environnement virtuel non trouvé à $VENV_ACTIVATE_ABS"
  read -p "Press Enter to close..."
  exit 1
fi

# Lancer le script python principal
if [ -f "$PROJECT_SCRIPT_ABS" ]; then
  python3 "$PROJECT_SCRIPT_ABS"
else
  echo "Erreur : script python introuvable à $PROJECT_SCRIPT_ABS"
  read -p "Press Enter to close..."
  exit 1
fi

read -p "Press Enter to close..."
