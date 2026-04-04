#!/bin/bash
set -e

echo "Installing dependencies..."
pip install --quiet -r requirements.txt

echo "Pre-downloading sentence-transformers model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Model cached.')
"

echo "Codespace ready. Run: python -m streamlit run streamlit_app.py"