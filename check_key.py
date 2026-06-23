import os

api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    print("API key not found")
else:
    print("API key found")
    