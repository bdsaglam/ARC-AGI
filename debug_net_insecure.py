import httpx
import sys

print("Testing INSECURE httpx connection...", file=sys.stderr)
try:
    client = httpx.Client(verify=False)
    resp = client.get("https://api.openai.com/v1/models", timeout=5)
    print(f"Success (OpenAI): {resp.status_code}", file=sys.stderr)
except Exception as e:
    print(f"Failed OpenAI: {e}", file=sys.stderr)

try:
    client = httpx.Client(verify=False)
    resp = client.get("https://api.anthropic.com", timeout=5)
    print(f"Success (Anthropic): {resp.status_code}", file=sys.stderr)
except Exception as e:
    print(f"Failed Anthropic: {e}", file=sys.stderr)
