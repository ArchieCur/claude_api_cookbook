from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import ToolParam
import json

# Load env variables and create client
client = Anthropic()
model = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku (fast and cost-effective)


# ============================================
# Helper Functions
# ============================================

def extract_structured_data(content, schema, schema_name):
    """Extract structured data from text using a tool schema.

    This function forces Claude to structure unstructured text according to
    a predefined schema. The tool is NEVER executed - we just use the schema
    for validation and extract the structured input.

    Args:
        content: The text content to extract structured data from
        schema: A tool schema (dict or ToolParam) defining the structure
        schema_name: The name of the schema tool

    Returns:
        dict: The structured data extracted from the content
    """
    messages = [{
        "role": "user",
        "content": content
    }]

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
        tools=[schema],
        tool_choice={"type": "tool", "name": schema_name}
    )

    # Extract structured data from tool_use block (never execute the tool)
    return response.content[0].input


# ============================================
# Example Schemas
# ============================================

# Example 1: Extract person information
person_schema = {
    "name": "extract_person",
    "description": "Extract person information from text",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's full name"
            },
            "age": {
                "type": "number",
                "description": "The person's age"
            },
            "occupation": {
                "type": "string",
                "description": "The person's job or profession"
            }
        },
        "required": ["name"]
    }
}

# Example 2: Extract article metadata
article_schema = {
    "name": "extract_article",
    "description": "Extract structured metadata from an article",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The article title"
            },
            "author": {
                "type": "string",
                "description": "The article author"
            },
            "key_insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of main takeaways from the article"
            },
            "category": {
                "type": "string",
                "description": "The article category or topic"
            }
        },
        "required": ["title", "key_insights"]
    }
}

# Example 3: Extract product information
product_schema = {
    "name": "extract_product",
    "description": "Extract product details from a description",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Product name"
            },
            "price": {
                "type": "number",
                "description": "Product price (numeric value only)"
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of product features"
            },
            "in_stock": {
                "type": "boolean",
                "description": "Whether the product is in stock"
            }
        },
        "required": ["name", "features"]
    }
}

# Example 4: Extract event information
event_schema = {
    "name": "extract_event",
    "description": "Extract event details from text",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_name": {
                "type": "string",
                "description": "Name of the event"
            },
            "date": {
                "type": "string",
                "description": "Event date"
            },
            "location": {
                "type": "string",
                "description": "Event location or venue"
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendees or participants"
            }
        },
        "required": ["event_name", "date"]
    }
}


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Extract Person Information")
    print("=" * 60)

    person_text = """
    Sarah Johnson is a 32-year-old software engineer who works at
    a major tech company. She specializes in machine learning and
    has been in the industry for 8 years.
    """

    person_data = extract_structured_data(person_text, person_schema, "extract_person")
    print(json.dumps(person_data, indent=2))

    print("\n" + "=" * 60)
    print("Example 2: Extract Article Metadata")
    print("=" * 60)

    article_text = """
    # The Future of AI in Healthcare
    By Dr. Emily Chen

    Artificial intelligence is transforming healthcare in remarkable ways.
    Machine learning algorithms can now detect diseases earlier than ever before.
    Personalized medicine is becoming a reality through AI-powered genomics.
    The integration of AI is reducing diagnostic errors significantly.
    """

    article_data = extract_structured_data(article_text, article_schema, "extract_article")
    print(json.dumps(article_data, indent=2))

    print("\n" + "=" * 60)
    print("Example 3: Extract Product Information")
    print("=" * 60)

    product_text = """
    Introducing the UltraBook Pro - now available for $1299!
    This laptop features a stunning 4K display, 32GB RAM, and
    a powerful M3 processor. It's incredibly lightweight at just
    2.5 pounds and offers 18 hours of battery life. Currently in stock.
    """

    product_data = extract_structured_data(product_text, product_schema, "extract_product")
    print(json.dumps(product_data, indent=2))

    print("\n" + "=" * 60)
    print("Example 4: Extract Event Information")
    print("=" * 60)

    event_text = """
    Join us for the Tech Summit 2025 on March 15th at the San Francisco
    Convention Center. Confirmed speakers include Sarah Chen, Mike Rodriguez,
    and Dr. Aisha Patel. Don't miss this exciting conference!
    """

    event_data = extract_structured_data(event_text, event_schema, "extract_event")
    print(json.dumps(event_data, indent=2))

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
This pattern is useful when you need to:
1. Extract structured data from unstructured text
2. Validate data against a specific schema
3. Convert free-form text into JSON objects
4. Parse information without complex regex or manual parsing

Key advantages:
- No tool execution needed (just schema validation)
- More reliable than prompt-based extraction
- Automatic type validation
- Easy to maintain and extend

When to use this vs traditional tools:
- Use this pattern: For data extraction and structuring
- Use traditional tools: For actual operations (API calls, calculations, etc.)
""")
