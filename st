#!/usr/bin/env python3
"""
/st

SwampThing CLI - Main entry point for all SwampThing commands.

Author: Forest Mars
Version: 0.2

Usage:
  st ingest                          # Run document ingestion
  st manage list                     # List clusters
  st manage docs <id> [--page N]     # List docs in cluster
  st manage summary <doc_id>         # Get document summary
  st manage rename <id> <name>       # Rename cluster
  st test query [question]           # Test query
  st test components                 # Test components
"""
__version__ = '0.2'
__author__ = 'Forest Mars'

import sys
import os
import subprocess
from pathlib import Path

# Get project root (where this script lives)
PROJECT_ROOT = Path(__file__).parent.absolute()

def run_command(script_path, args):
    """Run a Python script with uv."""
    cmd = ["uv", "run", "python", str(script_path)] + args
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == "ingest":
        run_command(PROJECT_ROOT / "src/ingest/ingest_documents.py", args)
    
    elif command == "manage":
        run_command(PROJECT_ROOT / "src/manage/manage_clusters.py", args)
    
    elif command == "test":
        if not args:
            print("Usage: st test [query|components]")
            sys.exit(1)
        
        test_type = args[0]
        test_args = args[1:]
        
        if test_type == "query":
            run_command(PROJECT_ROOT / "scripts/test_query.py", test_args)
        elif test_type == "components":
            run_command(PROJECT_ROOT / "scripts/test_components.py", test_args)
        else:
            print(f"Unknown test type: {test_type}")
            print("Available: query, components")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()