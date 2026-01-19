from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic


client = Anthropic()
model = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku (fast and cost-effective)


import json

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


#Generating dataset
def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
  {
    "task": "Description of task",
    "format": "python" or "json" or "regex",
    "solution_criteria": "Key criteria for evaluating the solution"
  },
  ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""

    #Parse JSON
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)

# Test the dataset generation
dataset = generate_dataset()
print(json.dumps(dataset, indent=2))

# Save to dataset.json
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
