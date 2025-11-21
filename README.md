# ARC-AGI

## Quickstart

```bash
git clone <repo-url>
cd ARC-AGI
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp config/api_keys.env.example config/api_keys.env  # then add your keys
python main.py
```

The `.venv/` directory is ignored by Git, so every contributor can maintain their own environment locally without polluting the repo. Add any third-party libraries you install to `requirements.txt` so others can reproduce the environment quickly.
