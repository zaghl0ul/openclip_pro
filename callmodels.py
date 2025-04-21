import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Fetch and print available Gemini models
for model in genai.list_models():
    print(f"Model Name: {model.name}")
    print(f"Supported methods: {model.supported_generation_methods}")
    print('-' * 40)
