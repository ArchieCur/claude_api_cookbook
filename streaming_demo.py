from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-sonnet-4-5-20250929"

# Helper function
def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

print("=== RAW EVENT STREAM ===")
print("This shows all streaming events:\n")

# Method 1: Raw event stream (see all events)
messages = []
add_user_message(messages, "Write a 1 sentence description of a fake database")

stream = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    stream=True
)

for event in stream:
    print(event)

print("\n\n=== SIMPLIFIED TEXT STREAM ===")
print("This shows only the text as it streams:\n")

# Method 2: Simplified text stream (just the text)
messages = []
add_user_message(messages, "Write a 1 sentence description of a fake database")

with client.messages.stream(
    model=model,
    max_tokens=1000,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print("\n\nDone!")
