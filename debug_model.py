import os
import sys
from openai import OpenAI
from src.config import get_api_keys, get_http_client

def debug_hello_world():
    print("Initializing OpenAI client...")
    openai_key, _, _ = get_api_keys()
    client = OpenAI(api_key=openai_key, http_client=get_http_client())

    model = "gpt-5.2-codex"
    effort = "none"
    
    print(f"Attempting to call model: {model} with reasoning effort: {effort}")

    try:
        kwargs = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello world. Just say hi."}]}],
            "timeout": 60,
            "stream": False,
        }
        
        # Add reasoning parameter if effort is not none (though for 'none' usually we omit it or pass 'none'?)
        # src/providers/openai_runner.py says: if self.reasoning_effort != "none": kwargs["reasoning"] = ...
        # So if effort is "none", we DON'T send reasoning param?
        # Let's try WITHOUT it first as per existing logic.
        
        # Actually, let's stick STRICTLY to what openai_runner.py does.
        if effort != "none":
            kwargs["reasoning"] = {"effort": effort}
            
        print(f"Sending request with kwargs: {kwargs}")
        
        response = client.responses.create(**kwargs)
        
        print("\nSUCCESS! Response received:")
        print(response)
        
    except Exception as e:
        print(f"\nFAILED with error:")
        print(e)

if __name__ == "__main__":
    debug_hello_world()
