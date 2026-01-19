from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
import json

client = Anthropic()
model = "claude-sonnet-4-5-20250929"

# Helper functions for managing chat history
def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)
    return message.content[0].text

# ============================================
# Structured Data Extraction Pattern
# ============================================

print("=== EXTRACTING CLEAN JSON ===\n")

messages = []
add_user_message(messages, "Generate a very short event bridge rule as json")
add_assistant_message(messages, "```json")

# Get JSON between code fences
text = chat(messages, stop_sequences=["```"])

print("Raw response:")
print(text)
print("\n" + "="*50 + "\n")

# Clean up and parse the JSON
clean_json = json.loads(text.strip())

print("Parsed JSON (Python dict):")
print(clean_json)
print("\n" + "="*50 + "\n")

print("Accessing JSON fields:")
for key, value in clean_json.items():
    print(f"  {key}: {value}")
