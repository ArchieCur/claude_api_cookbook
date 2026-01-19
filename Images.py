from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load env variables and create client
client = Anthropic()
model = "claude-sonnet-4-5"  # Vision requires Sonnet 3.5+ or Opus 3+


# ============================================
# Helper Functions
# ============================================

def add_user_message(messages, message):
    """Add a user message to the conversation history."""
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    """Add an assistant message to the conversation history."""
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message,
    }
    messages.append(assistant_message)


def chat(
    messages,
    system=None,
    temperature=1.0,
    stop_sequences=[],
    tools=None,
    thinking=False,
    thinking_budget=1024,
):
    """Make an API call to Claude with optional images, tools, and thinking."""
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if thinking:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    """Extract all text content from a message."""
    return "\n".join([block.text for block in message.content if block.type == "text"])


# ============================================
# Image Handling Functions
# ============================================

def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a supported image format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    ext = Path(image_path).suffix.lower()
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported image format: {ext}. Supported: {valid_extensions}")

    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type for an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Media type string (e.g., "image/jpeg", "image/png")

    Raises:
        ValueError: If file extension is not supported
    """
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }

    if ext not in media_types:
        raise ValueError(f"Unsupported image extension: {ext}")

    return media_types[ext]


def create_image_block(image_path: str) -> Dict[str, Any]:
    """Create an image content block for Claude API.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary representing an image content block
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": get_image_media_type(image_path),
            "data": encode_image(image_path),
        },
    }


def create_message_with_image(text: str, image_path: str) -> List[Dict[str, Any]]:
    """Create a message content list with text and a single image.

    Args:
        text: The text prompt/question
        image_path: Path to the image file

    Returns:
        List of content blocks (text + image)
    """
    return [
        {"type": "text", "text": text},
        create_image_block(image_path),
    ]


def create_message_with_images(
    text: str,
    image_paths: List[str],
    interleave: bool = False
) -> List[Dict[str, Any]]:
    """Create a message content list with text and multiple images.

    Args:
        text: The text prompt/question
        image_paths: List of paths to image files
        interleave: If True, place text between images; if False, text first then all images

    Returns:
        List of content blocks (text + images)
    """
    content = []

    if interleave:
        # Interleave text and images
        content.append({"type": "text", "text": text})
        for image_path in image_paths:
            content.append(create_image_block(image_path))
    else:
        # Text first, then all images
        content.append({"type": "text", "text": text})
        for image_path in image_paths:
            content.append(create_image_block(image_path))

    return content


def create_image_with_caption(image_path: str, caption: str) -> List[Dict[str, Any]]:
    """Create content blocks for an image with a caption.

    Useful for providing context about what each image represents.

    Args:
        image_path: Path to the image file
        caption: Caption/description for the image

    Returns:
        List of content blocks (caption text + image)
    """
    return [
        {"type": "text", "text": caption},
        create_image_block(image_path),
    ]


def analyze_image(image_path: str, prompt: str) -> str:
    """Analyze a single image with a prompt.

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image

    Returns:
        Claude's response text
    """
    messages = []
    content = create_message_with_image(prompt, image_path)
    add_user_message(messages, content)

    response = chat(messages)
    return text_from_message(response)


def analyze_images(
    image_paths: List[str],
    prompt: str,
    with_captions: Optional[List[str]] = None
) -> str:
    """Analyze multiple images with a prompt.

    Args:
        image_paths: List of paths to image files
        prompt: Question or instruction about the images
        with_captions: Optional captions for each image

    Returns:
        Claude's response text
    """
    messages = []
    content = []

    if with_captions and len(with_captions) == len(image_paths):
        # Add captioned images
        for image_path, caption in zip(image_paths, with_captions):
            content.extend(create_image_with_caption(image_path, caption))
        # Add the main prompt at the end
        content.append({"type": "text", "text": prompt})
    else:
        # Standard format: prompt + images
        content = create_message_with_images(prompt, image_paths)

    add_user_message(messages, content)

    response = chat(messages)
    return text_from_message(response)


def compare_images(image_path1: str, image_path2: str, comparison_prompt: str) -> str:
    """Compare two images.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        comparison_prompt: What to compare or analyze

    Returns:
        Claude's comparison response
    """
    return analyze_images(
        [image_path1, image_path2],
        comparison_prompt,
        with_captions=["Image 1:", "Image 2:"]
    )


# ============================================
# Common Image Analysis Tasks
# ============================================

def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image (OCR).

    Args:
        image_path: Path to the image file

    Returns:
        Extracted text
    """
    prompt = "Please extract all visible text from this image. Provide only the text content, maintaining the original structure and formatting as much as possible."
    return analyze_image(image_path, prompt)


def describe_image(image_path: str, detail_level: str = "medium") -> str:
    """Generate a description of an image.

    Args:
        image_path: Path to the image file
        detail_level: "brief", "medium", or "detailed"

    Returns:
        Image description
    """
    prompts = {
        "brief": "Describe this image in one sentence.",
        "medium": "Describe this image in 2-3 sentences, covering the main elements and any notable details.",
        "detailed": "Provide a detailed description of this image, including all visible elements, colors, composition, and any text or notable features.",
    }

    prompt = prompts.get(detail_level, prompts["medium"])
    return analyze_image(image_path, prompt)


def analyze_chart(image_path: str) -> str:
    """Analyze a chart or graph.

    Args:
        image_path: Path to the chart/graph image

    Returns:
        Analysis of the chart
    """
    prompt = """Analyze this chart or graph and provide:
1. Type of chart (bar, line, pie, scatter, etc.)
2. Title and axis labels if visible
3. Key data points and trends
4. Main insights or patterns
5. Any notable outliers or anomalies"""

    return analyze_image(image_path, prompt)


def analyze_document(image_path: str) -> str:
    """Analyze a document image.

    Args:
        image_path: Path to the document image

    Returns:
        Document analysis with extracted information
    """
    prompt = """Analyze this document and provide:
1. Document type (invoice, receipt, form, etc.)
2. Key information (dates, amounts, names, etc.)
3. All visible text in a structured format
4. Any notable features or observations"""

    return analyze_image(image_path, prompt)


def answer_question_about_image(image_path: str, question: str) -> str:
    """Answer a specific question about an image.

    Args:
        image_path: Path to the image file
        question: Specific question to answer

    Returns:
        Answer to the question
    """
    return analyze_image(image_path, question)


# ============================================
# Advanced Multi-Image Workflows
# ============================================

def analyze_image_sequence(
    image_paths: List[str],
    analysis_type: str = "progression"
) -> str:
    """Analyze a sequence of images.

    Args:
        image_paths: List of paths to images in sequence
        analysis_type: "progression", "comparison", or "summary"

    Returns:
        Analysis of the image sequence
    """
    prompts = {
        "progression": "These images show a progression or sequence. Describe what changes between each image and identify any patterns or trends.",
        "comparison": "Compare these images side by side. What are the similarities and differences?",
        "summary": "These images are related. Provide an overall summary of what they show collectively.",
    }

    prompt = prompts.get(analysis_type, prompts["progression"])

    captions = [f"Image {i+1}:" for i in range(len(image_paths))]

    return analyze_images(image_paths, prompt, with_captions=captions)


def create_image_report(image_path: str, report_type: str = "comprehensive") -> str:
    """Generate a comprehensive report about an image.

    Args:
        image_path: Path to the image file
        report_type: "comprehensive", "technical", or "summary"

    Returns:
        Detailed report about the image
    """
    prompts = {
        "comprehensive": """Provide a comprehensive analysis of this image including:
1. Overall description and subject matter
2. Visual elements (colors, composition, lighting)
3. Any text or data visible
4. Quality and technical aspects
5. Context and potential use cases
6. Notable features or points of interest""",

        "technical": """Provide a technical analysis of this image:
1. Apparent resolution and quality
2. Color palette and dominant colors
3. Composition and framing
4. Any technical issues or artifacts
5. Suitability for different purposes""",

        "summary": "Provide a brief executive summary of this image's content and purpose.",
    }

    prompt = prompts.get(report_type, prompts["comprehensive"])
    return analyze_image(image_path, prompt)


# ============================================
# Multi-Turn Image Conversations
# ============================================

def image_conversation(image_path: str, initial_prompt: str) -> Dict[str, Any]:
    """Start a multi-turn conversation about an image.

    Args:
        image_path: Path to the image file
        initial_prompt: First question or instruction

    Returns:
        Dictionary containing messages list and first response
    """
    messages = []
    content = create_message_with_image(initial_prompt, image_path)
    add_user_message(messages, content)

    response = chat(messages)
    add_assistant_message(messages, response)

    return {
        "messages": messages,
        "response": text_from_message(response),
    }


def continue_image_conversation(
    conversation_state: Dict[str, Any],
    follow_up_prompt: str
) -> Dict[str, Any]:
    """Continue a conversation about an image.

    Args:
        conversation_state: State from previous conversation turn
        follow_up_prompt: Next question or instruction

    Returns:
        Updated conversation state with new response
    """
    messages = conversation_state["messages"]
    add_user_message(messages, follow_up_prompt)

    response = chat(messages)
    add_assistant_message(messages, response)

    return {
        "messages": messages,
        "response": text_from_message(response),
    }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Image Analysis Examples")
    print("=" * 60)
    print("""
NOTE: These examples require actual image files to run.
Replace the placeholder paths with your own images.

Supported formats: JPEG, PNG, GIF, WebP
""")

    print("\n" + "=" * 60)
    print("Example 1: Analyze a Single Image")
    print("=" * 60)

    """
    # Uncomment and replace with your image path
    image_path = "path/to/your/image.jpg"

    description = describe_image(image_path, detail_level="detailed")
    print("Description:")
    print(description)
    """

    print("\n" + "=" * 60)
    print("Example 2: Extract Text from Image (OCR)")
    print("=" * 60)

    """
    # Uncomment and replace with your image path
    document_image = "path/to/document.png"

    extracted_text = extract_text_from_image(document_image)
    print("Extracted Text:")
    print(extracted_text)
    """

    print("\n" + "=" * 60)
    print("Example 3: Analyze a Chart or Graph")
    print("=" * 60)

    """
    # Uncomment and replace with your chart image path
    chart_image = "path/to/chart.png"

    chart_analysis = analyze_chart(chart_image)
    print("Chart Analysis:")
    print(chart_analysis)
    """

    print("\n" + "=" * 60)
    print("Example 4: Compare Two Images")
    print("=" * 60)

    """
    # Uncomment and replace with your image paths
    before_image = "path/to/before.jpg"
    after_image = "path/to/after.jpg"

    comparison = compare_images(
        before_image,
        after_image,
        "What are the main differences between these two images?"
    )
    print("Comparison:")
    print(comparison)
    """

    print("\n" + "=" * 60)
    print("Example 5: Analyze Multiple Images")
    print("=" * 60)

    """
    # Uncomment and replace with your image paths
    images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]

    analysis = analyze_images(
        images,
        "What story do these images tell together?",
        with_captions=["Scene 1:", "Scene 2:", "Scene 3:"]
    )
    print("Multi-Image Analysis:")
    print(analysis)
    """

    print("\n" + "=" * 60)
    print("Example 6: Image Sequence Analysis")
    print("=" * 60)

    """
    # Uncomment and replace with your image paths
    sequence = [
        "path/to/step1.jpg",
        "path/to/step2.jpg",
        "path/to/step3.jpg",
        "path/to/step4.jpg",
    ]

    progression = analyze_image_sequence(sequence, analysis_type="progression")
    print("Sequence Analysis:")
    print(progression)
    """

    print("\n" + "=" * 60)
    print("Example 7: Multi-Turn Conversation About an Image")
    print("=" * 60)

    """
    # Uncomment and replace with your image path
    image = "path/to/complex_image.jpg"

    # Start conversation
    conv = image_conversation(
        image,
        "What is the main subject of this image?"
    )
    print("Turn 1:", conv["response"])

    # Follow-up question
    conv = continue_image_conversation(
        conv,
        "What colors are dominant in this image?"
    )
    print("Turn 2:", conv["response"])

    # Another follow-up
    conv = continue_image_conversation(
        conv,
        "Based on what you see, what might be the context or setting?"
    )
    print("Turn 3:", conv["response"])
    """

    print("\n" + "=" * 60)
    print("Example 8: Custom Analysis Prompt")
    print("=" * 60)

    """
    # Uncomment and replace with your image path
    image = "path/to/photo.jpg"

    custom_prompt = \"\"\"
    Analyze this image for:
    1. Any safety hazards visible
    2. Accessibility features or barriers
    3. Maintenance needs or issues
    4. Overall condition rating (1-10)

    Provide specific details for each category.
    \"\"\"

    analysis = analyze_image(image, custom_prompt)
    print("Custom Analysis:")
    print(analysis)
    """

    print("\n" + "=" * 60)
    print("Example 9: Analyze Document with Structured Output")
    print("=" * 60)

    """
    # Uncomment and replace with your document image path
    receipt_image = "path/to/receipt.jpg"

    structured_prompt = \"\"\"
    Extract information from this receipt and format as JSON:
    {
        "merchant": "...",
        "date": "...",
        "total": "...",
        "items": ["...", "..."],
        "payment_method": "..."
    }
    \"\"\"

    result = analyze_image(receipt_image, structured_prompt)
    print("Structured Data:")
    print(result)
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Image Analysis with Claude:

SUPPORTED FORMATS:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

IMAGE SIZE LIMITS:
- Maximum file size: 5MB per image (recommended: <3MB)
- Maximum dimensions: 8000 x 8000 pixels
- Images are automatically resized if needed
- For best performance: 1568 pixels on longest side

ENCODING REQUIREMENTS:
- Images must be base64 encoded
- Include proper media type (image/jpeg, image/png, etc.)
- Use standard_b64encode (not urlsafe)

MESSAGE STRUCTURE:

Single Image:
[
    {"type": "text", "text": "Your prompt"},
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "<base64_string>"
        }
    }
]

Multiple Images:
[
    {"type": "text", "text": "Your prompt"},
    {"type": "image", "source": {...}},
    {"type": "image", "source": {...}},
    ...
]

Images with Captions:
[
    {"type": "text", "text": "Caption 1"},
    {"type": "image", "source": {...}},
    {"type": "text", "text": "Caption 2"},
    {"type": "image", "source": {...}},
    {"type": "text", "text": "Main prompt"}
]

COMMON USE CASES:

1. OCR / Text Extraction:
   - Documents, receipts, forms
   - Screenshots with text
   - Handwritten notes (mixed results)

2. Visual Analysis:
   - Photos, artwork, designs
   - Product images
   - Scene understanding

3. Charts & Graphs:
   - Data visualization analysis
   - Trend identification
   - Statistical insights

4. Document Analysis:
   - Invoice processing
   - Form data extraction
   - ID/license reading

5. Image Comparison:
   - Before/after comparisons
   - Quality assessment
   - Change detection

6. Multi-Image Tasks:
   - Sequence analysis
   - Step-by-step guides
   - Progress tracking

BEST PRACTICES:

Prompt Engineering:
✓ Be specific about what you want to extract/analyze
✓ Structure requests with numbered steps for complex tasks
✓ Use captions to clarify which image is which
✓ Ask for structured output (JSON, tables, lists) when needed

Image Quality:
✓ Use clear, well-lit images
✓ Ensure text is legible (300 DPI for documents)
✓ Crop to relevant area before sending
✓ Optimize file size (compress without losing quality)

Performance:
✓ Send multiple images in one request (not separate requests)
✓ Use appropriate model (Sonnet for balance, Opus for best quality)
✓ Consider using Haiku for simple OCR tasks (cost savings)
✓ Batch similar image analysis tasks

Error Handling:
✓ Validate image format before encoding
✓ Check file existence before processing
✓ Handle encoding errors gracefully
✓ Provide fallback for unsupported formats

COST CONSIDERATIONS:

Token Usage:
- Images are converted to tokens for pricing
- Token count depends on image size (not file size)
- Larger images = more tokens
- Multiple images multiply token costs

Cost Optimization:
1. Resize images to optimal dimensions (~1568px longest side)
2. Use appropriate compression
3. Send only necessary images
4. Use Haiku for simple tasks
5. Batch requests when possible

Approximate Token Costs:
- Small image (400x400): ~200 tokens
- Medium image (800x800): ~800 tokens
- Large image (1568x1568): ~1600 tokens
- Multiple images: sum of individual token costs

LIMITATIONS:

What Claude CAN Do:
✓ Describe images in detail
✓ Extract visible text (OCR)
✓ Analyze charts and data visualizations
✓ Compare multiple images
✓ Identify objects, people, scenes
✓ Answer questions about image content
✓ Provide structured data extraction

What Claude CANNOT Do:
✗ Identify specific individuals (privacy protection)
✗ Generate or edit images
✗ Access EXIF metadata
✗ Perform pixel-perfect measurements
✗ Guarantee 100% OCR accuracy
✗ Process video or animated content
✗ Read heavily distorted or damaged text

MULTI-TURN CONVERSATIONS:

Pattern:
1. Send image in first message
2. Claude analyzes and responds
3. Follow-up questions reference the same image
4. No need to resend image in subsequent turns
5. Image context is maintained throughout conversation

Benefits:
- Saves tokens (image sent once)
- Natural conversation flow
- Progressive refinement of analysis
- Ask clarifying questions

PRODUCTION TIPS:

1. Validation:
   - Verify image format and size before processing
   - Handle missing files gracefully
   - Provide clear error messages to users

2. User Experience:
   - Show preview of images being analyzed
   - Provide progress indicators for long requests
   - Allow users to crop/adjust before sending
   - Display confidence levels when extracting data

3. Security:
   - Validate image content (check for malicious files)
   - Sanitize extracted text (prevent injection attacks)
   - Consider privacy implications (PII in images)
   - Implement rate limiting for image uploads

4. Monitoring:
   - Track token usage per image type
   - Monitor accuracy of extraction tasks
   - Log errors for improvement
   - A/B test prompt variations

5. Caching:
   - Cache image analysis results
   - Store extracted data for reuse
   - Use prompt caching for repeated similar requests
   - Implement deduplication for identical images

INTEGRATION PATTERNS:

Pattern 1: Simple OCR Service
def ocr_service(image_path):
    try:
        return extract_text_from_image(image_path)
    except Exception as e:
        return {"error": str(e)}

Pattern 2: Batch Image Processing
def process_images_batch(image_paths, prompt):
    results = []
    for path in image_paths:
        result = analyze_image(path, prompt)
        results.append({"image": path, "analysis": result})
    return results

Pattern 3: Image + Tools
# Combine image analysis with tool use
# e.g., analyze chart, then use calculator tool for computations

Pattern 4: Structured Data Pipeline
# Image → Claude (extract) → Validate → Database
def document_to_database(image_path):
    extracted = analyze_document(image_path)
    validated = validate_extracted_data(extracted)
    save_to_database(validated)

ADVANCED TECHNIQUES:

1. Prompt Chaining:
   First: "Describe this image"
   Then: "Based on that description, what safety concerns exist?"

2. Few-Shot with Images:
   Provide example images + desired outputs
   Then send target image for same analysis

3. Structured Output:
   Request JSON/XML format
   Use schema in prompt
   Validate response format

4. Multi-Modal RAG:
   Index image descriptions
   Retrieve relevant images
   Combine with text context

5. Quality Assessment:
   Ask Claude to rate confidence
   Request uncertainty indicators
   Verify critical extractions
""")
