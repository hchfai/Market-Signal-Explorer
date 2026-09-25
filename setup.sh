#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env -- add your NewsAPI key before running the app (optional)."
fi

echo ""
echo "Setup complete."
echo "Next time, activate with: source .venv/bin/activate"
echo "Then run:                 streamlit run app.py"
