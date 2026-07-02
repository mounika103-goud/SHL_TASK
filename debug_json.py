import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; import os; load_dotenv()
from google import genai
from google.genai import types
import json

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
config = types.GenerateContentConfig(temperature=0.1, response_mime_type='application/json')
r = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Output JSON with keys: action (string), reply (string). Set action to clarify.',
    config=config
)
print("Raw output repr:", repr(r.text[:500]))
print()
try:
    parsed = json.loads(r.text)
    print("Parsed OK:", parsed)
except Exception as e:
    print("Parse error:", e)
