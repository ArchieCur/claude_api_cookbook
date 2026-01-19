from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
import json
import re
import ast
from statistics import mean

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


# Code Validation Functions

def validate_json(text):
    """Validates if text is valid JSON. Returns 10 if valid, 0 if not."""
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    """Validates if text is valid Python syntax. Returns 10 if valid, 0 if not."""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    """Validates if text is a valid regex. Returns 10 if valid, 0 if not."""
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response, test_case):
    """Grades the syntax based on the expected format."""
    format_type = test_case["format"]
    if format_type == "json":
        return validate_json(response)
    elif format_type == "python":
        return validate_python(response)
    else:
        return validate_regex(response)


# Evaluation Pipeline Functions

def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}

* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explanation
"""

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```code")
    output = chat(messages, stop_sequences=["```"])
    return output


def grade_by_model(test_case, output):
    """Grades output using Claude as a judge."""
    eval_prompt = f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Solution to Evaluate:
<solution>
{output}
</solution>

Criteria you should use to evaluate the solution:
<criteria>
{test_case["solution_criteria"]}
</criteria>

Output Format
Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with JSON. Keep your response concise and direct.
Example response shape:
{{
    "strengths": string[],
    "weaknesses": string[],
    "reasoning": string,
    "score": number
}}
"""

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


def run_test_case(test_case):
    """Calls run_prompt, then grades the result using both model and syntax grading."""
    output = run_prompt(test_case)

    # Model-based grading
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    # Syntax grading
    syntax_score = grade_syntax(output, test_case)

    # Average the two scores
    score = (model_score + syntax_score) / 2

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning,
        "model_score": model_score,
        "syntax_score": syntax_score
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score}")

    return results


# ============================================
# Run Evaluation Pipeline
# ============================================

if __name__ == "__main__":
    print("Running code-based evaluation pipeline...")
    print("=" * 60)

    # Load dataset from JSON file
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    # Run evaluation on all test cases
    results = run_eval(dataset)

    # Display detailed results
    print("\nDetailed Evaluation Results:")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\nTest Case {i}:")
        print(f"Task: {result['test_case']['task']}")
        print(f"Format: {result['test_case']['format']}")
        print(f"Model Score: {result['model_score']}/10")
        print(f"Syntax Score: {result['syntax_score']}/10")
        print(f"Final Score: {result['score']}/10")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Output:\n{result['output']}")
        print("-" * 60)

    # Calculate and display average score
    avg_score = mean([r['score'] for r in results])
    print(f"\nAverage Score: {avg_score}/10")
