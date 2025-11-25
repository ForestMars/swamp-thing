#!/bin/bash
# --- Fix: Get the absolute directory of the currently running script ---
# Resolves the path of the script, removing '..' and resolving links, 
# ensuring the correct file location is found regardless of where it's executed from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

USER_=$(whoami)
echo "Current OS User: $USER_"

# --- CORRECTED COMMAND ---
# Executes the SQL file using the dynamically determined absolute path ($SCRIPT_DIR)
# -U: Authenticates as your current OS user (superuser).
# -d: Connects to the existing 'postgres' maintenance database.
# -f: Executes the SQL file located at the guaranteed correct path.
psql -U $USER_ -d postgres -f "$SCRIPT_DIR/rag_user.sql"
