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

def chat(messages, system=None, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text

# Define system prompt
system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""

# Interactive chat loop
messages = []

print("Math Tutor Chat! (Type 'quit' to exit)")
print("-" * 50)

while True:
    user_input = input("> ")
    print(">", user_input)

    # Exit condition
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break

    # Add user message to history
    add_user_message(messages, user_input)

    # Get Claude's response with system prompt
    response = chat(messages, system=system_prompt)

    # Print Claude's response
    print(f"Tutor: {response}\n")

    # Add Claude's response to history
    add_assistant_message(messages, response)
