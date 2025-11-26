import httpx
import sys
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()

print("Testing httpx connection to google.com...", file=sys.stderr)
try:
    resp = httpx.get("https://www.google.com", timeout=5, verify=certifi.where())
    print(f"Success: {resp.status_code}", file=sys.stderr)
except Exception as e:
    print(f"Failed google: {e}", file=sys.stderr)

print("Testing httpx connection to api.openai.com...", file=sys.stderr)
try:
    resp = httpx.get("https://api.openai.com/v1/models", timeout=5, verify=certifi.where()) 
    print(f"Success (OpenAI): {resp.status_code}", file=sys.stderr)
except Exception as e:
    print(f"Failed OpenAI: {e}", file=sys.stderr)

print("Testing httpx connection to api.anthropic.com...", file=sys.stderr)
try:
    resp = httpx.get("https://api.anthropic.com", timeout=5, verify=certifi.where()) 
    print(f"Success (Anthropic): {resp.status_code}", file=sys.stderr)
except Exception as e:
    print(f"Failed Anthropic: {e}", file=sys.stderr)
