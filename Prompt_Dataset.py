from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
import json
from Prompting import PromptEvaluator

# Client already initialized in Prompting.py

# Create an instance of PromptEvaluator
# Set max_concurrent_tasks=1 to avoid rate limit errors
evaluator = PromptEvaluator(max_concurrent_tasks=1)

# Generate dataset for meal plan prompt
dataset = evaluator.generate_dataset(
    # Describe the purpose or goal of the prompt you're trying to test
    task_description="Generate a 1-day meal plan for an athlete based on their height, weight, goal, and dietary restrictions",

    # Describe the different inputs that your prompt requires
    prompt_inputs_spec={
        "height": "The athlete's height in inches or cm",
        "weight": "The athlete's weight in pounds or kg",
        "goal": "The athlete's fitness goal (e.g., weight loss, muscle gain, maintenance, endurance)",
        "dietary_restrictions": "Any dietary restrictions or preferences (e.g., vegetarian, vegan, gluten-free, dairy-free, allergies)"
    },

    # Where to write the generated dataset
    output_file="Prompt_dataset.json",

    # Number of test cases to generate (recommend keeping this low if you're getting rate limit errors)
    num_cases=3,
)

print("\nDataset generated successfully!")
print(json.dumps(dataset, indent=2))
