from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-sonnet-4-5-20250929"

# Helper functions
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


print("=== DEMO 1: MESSAGE PREFILLING ===")
print("Prefilling guides Claude's response direction\n")

messages = []
add_user_message(messages, "Is tea or coffee better at breakfast?")
add_assistant_message(messages, "Coffee is better because")
answer = chat(messages)

print("User: Is tea or coffee better at breakfast?")
print(f"Assistant (prefilled): Coffee is better because{answer}\n")


print("\n=== DEMO 2: STOP SEQUENCES ===")
print("Stop sequences make Claude stop at specific strings\n")

messages = []
add_user_message(messages, "Count from 1 to 10")
answer = chat(messages, stop_sequences=["5"])

print("User: Count from 1 to 10")
print(f"Assistant (stops at '5'): {answer}")


print("\n\n=== DEMO 3: COMBINING BOTH ===")
print("Prefill + stop sequences for precise control\n")

messages = []
add_user_message(messages, "List 5 programming languages")
add_assistant_message(messages, "Here are 5 programming languages:\n1.")
answer = chat(messages, stop_sequences=["6."])

print("User: List 5 programming languages")
print(f"Assistant: Here are 5 programming languages:\n1.{answer}")
