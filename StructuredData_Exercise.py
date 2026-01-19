from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic


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
        "temperature": temperature,
        "stop_sequences": stop_sequences
    }

    if system:
        params["system"] = system

    
    message = client.messages.create(**params)
    return message.content[0].text

# ============================================
# Structured Data Extraction Pattern
# ============================================

print("=== EXTRACTING AWS CLI COMMANDS ===\n")

messages = []

prompt = """Generate three different sample AWS CLI commands. Each should be very short."""

add_user_message(messages, prompt)
add_assistant_message(messages, "Here are all three commands in a single block without any comments:\n```bash")

# Get code from between code fences
text = chat(messages, stop_sequences=["```"])
# Clean up 
commands = text.strip()



print(commands)


