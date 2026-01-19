from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
import json

client = Anthropic()
model = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku (fast and cost-effective)

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


# Evaluation Pipeline Functions

def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Implement actual grading logic
    # For now, score is hardcoded to 10
    score = 10

    return {
        "output": output,
        "test_case": test_case,
        "score": score
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    return results


# ============================================
# Run Evaluation Pipeline
# ============================================

if __name__ == "__main__":
    print("Running evaluation pipeline...")
    print("=" * 60)

    # Load dataset from JSON file
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    # Run evaluation on all test cases
    results = run_eval(dataset)

    # Display results
    print("\nEvaluation Results:")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i}:")
        print(f"Task: {result['test_case']['task']}")
        print(f"Score: {result['score']}/10")
        print(f"Output:\n{result['output']}")
        print("-" * 60)

    # Calculate and display average score
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score}/10")
