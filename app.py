import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))

print("Available models for generateContent:")
print("-" * 50)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✓ {m.name}")
        print(f"  Display: {m.display_name}")
        print(f"  Description: {m.description[:100] if m.description else 'N/A'}")
        print()