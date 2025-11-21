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
# Export secrets (options below)
# Option A: export variables manually
export OPENAI_API_KEY=...
# Option B: source the config with auto-export in the shell session
set -a && source config/api_keys.env && set +a
python main.py data/arc-agi-2-training/00576224.json data/arc-agi-2-training/007bbfb7.json --reasoning none
```

The `.venv/` directory is ignored by Git, so every contributor can maintain their own environment locally without polluting the repo. Add any third-party libraries you install to `requirements.txt` so others can reproduce the environment quickly.

`main.py` loads one or more ARC-AGI puzzle files, packages the training examples plus each test input into an OpenAI prompt, requests a completion from `gpt-5.1`, and parses the returned grid. The predicted grid is compared against the ground-truth test output from the JSON file, and the script prints PASS/FAIL for each test case. Use `--reasoning` to select the effort level (supported by `gpt-5.1`: `none`, `low`, `medium`, or `high`). Ensure `OPENAI_API_KEY` is exported (see example commands above) before running it; otherwise the script will exit with an error.
