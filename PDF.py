from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional: For PDF type detection (install with: pip install PyPDF2)
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Load env variables and create client
client = Anthropic()
model = "claude-sonnet-4-5"  # PDF processing requires Sonnet 3.5+ or Opus 3+


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
    """Make an API call to Claude with optional PDFs, tools, and thinking."""
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
# PDF Handling Functions
# ============================================

def encode_pdf(pdf_path: str) -> str:
    """Encode a PDF file to base64 string.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Base64 encoded string of the PDF

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If file is not a PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Check file extension
    ext = Path(pdf_path).suffix.lower()
    if ext != '.pdf':
        raise ValueError(f"File must be a PDF. Got: {ext}")

    with open(pdf_path, "rb") as pdf_file:
        return base64.standard_b64encode(pdf_file.read()).decode("utf-8")


def get_pdf_size(pdf_path: str) -> int:
    """Get the size of a PDF file in bytes.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        File size in bytes
    """
    return os.path.getsize(pdf_path)


def validate_pdf_size(pdf_path: str, max_size_mb: int = 32) -> bool:
    """Validate PDF file size is within limits.

    Args:
        pdf_path: Path to the PDF file
        max_size_mb: Maximum allowed size in MB (default: 32MB)

    Returns:
        True if valid, False otherwise
    """
    size_bytes = get_pdf_size(pdf_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb <= max_size_mb


# ============================================
# PDF Type Detection
# ============================================

def detect_pdf_type(pdf_path: str) -> Dict[str, Any]:
    """Detect if a PDF is text-based or image-based (scanned).

    Requires PyPDF2 library (optional). Install with: pip install PyPDF2

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with detection results:
        {
            "type": "text-based" | "image-based" | "mixed" | "unknown",
            "confidence": "high" | "medium" | "low",
            "text_pages": int,
            "image_pages": int,
            "total_pages": int,
            "has_extractable_text": bool,
            "recommendation": str
        }
    """
    if not PYPDF2_AVAILABLE:
        return {
            "type": "unknown",
            "confidence": "low",
            "text_pages": 0,
            "image_pages": 0,
            "total_pages": 0,
            "has_extractable_text": False,
            "recommendation": "Install PyPDF2 for PDF type detection: pip install PyPDF2"
        }

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            text_pages = 0
            image_pages = 0

            # Sample up to 5 pages for detection (first, middle, last)
            sample_pages = []
            if total_pages <= 5:
                sample_pages = list(range(total_pages))
            else:
                sample_pages = [
                    0,  # First page
                    total_pages // 4,  # Quarter
                    total_pages // 2,  # Middle
                    3 * total_pages // 4,  # Three-quarters
                    total_pages - 1  # Last page
                ]

            for page_num in sample_pages:
                page = reader.pages[page_num]
                text = page.extract_text().strip()

                # Consider a page "text-based" if it has substantial extractable text
                if len(text) > 50:  # More than 50 characters
                    text_pages += 1
                else:
                    image_pages += 1

            # Determine PDF type based on samples
            if text_pages == 0:
                pdf_type = "image-based"
                confidence = "high"
                recommendation = "This appears to be a scanned/image-based PDF. Consider using OCR preprocessing for better accuracy and lower token costs."
            elif image_pages == 0:
                pdf_type = "text-based"
                confidence = "high"
                recommendation = "Perfect! This is a text-based PDF and will work optimally with Claude."
            else:
                pdf_type = "mixed"
                confidence = "medium"
                recommendation = "This PDF contains both text and scanned pages. Claude will handle it, but results may vary by page."

            return {
                "type": pdf_type,
                "confidence": confidence,
                "text_pages": text_pages,
                "image_pages": image_pages,
                "total_pages": total_pages,
                "has_extractable_text": text_pages > 0,
                "recommendation": recommendation
            }

    except Exception as e:
        return {
            "type": "unknown",
            "confidence": "low",
            "text_pages": 0,
            "image_pages": 0,
            "total_pages": 0,
            "has_extractable_text": False,
            "recommendation": f"Error detecting PDF type: {str(e)}"
        }


def analyze_pdf_quality(pdf_path: str) -> Dict[str, Any]:
    """Analyze PDF quality and suitability for Claude processing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with quality analysis and recommendations
    """
    quality_report = {
        "file_path": pdf_path,
        "file_size_mb": round(get_pdf_size(pdf_path) / (1024 * 1024), 2),
        "within_size_limit": validate_pdf_size(pdf_path),
        "issues": [],
        "warnings": [],
        "recommendations": [],
        "overall_score": "good"  # good, acceptable, poor
    }

    # Check file size
    if not quality_report["within_size_limit"]:
        quality_report["issues"].append("File exceeds 32MB limit")
        quality_report["recommendations"].append("Compress PDF or split into smaller files")
        quality_report["overall_score"] = "poor"
    elif quality_report["file_size_mb"] > 25:
        quality_report["warnings"].append("File is close to 32MB limit")
        quality_report["recommendations"].append("Consider optimizing PDF size for faster processing")

    # Detect PDF type
    pdf_type_info = detect_pdf_type(pdf_path)
    quality_report["pdf_type"] = pdf_type_info["type"]
    quality_report["type_confidence"] = pdf_type_info["confidence"]

    if pdf_type_info["type"] == "image-based":
        quality_report["warnings"].append("PDF appears to be scanned/image-based")
        quality_report["recommendations"].append(pdf_type_info["recommendation"])
        if quality_report["overall_score"] == "good":
            quality_report["overall_score"] = "acceptable"
    elif pdf_type_info["type"] == "mixed":
        quality_report["warnings"].append("PDF contains mixed content (text + images)")
        quality_report["recommendations"].append("Results may vary by page")
    elif pdf_type_info["type"] == "text-based":
        quality_report["recommendations"].append("Optimal PDF format for Claude")

    # Check if file exists and is readable
    if not os.path.exists(pdf_path):
        quality_report["issues"].append("File not found")
        quality_report["overall_score"] = "poor"

    # Overall assessment
    if len(quality_report["issues"]) > 0:
        quality_report["overall_score"] = "poor"
    elif len(quality_report["warnings"]) > 2:
        quality_report["overall_score"] = "acceptable"

    return quality_report


def print_pdf_quality_report(pdf_path: str):
    """Print a formatted quality report for a PDF.

    Args:
        pdf_path: Path to the PDF file
    """
    report = analyze_pdf_quality(pdf_path)

    print("=" * 60)
    print("PDF QUALITY REPORT")
    print("=" * 60)
    print(f"File: {report['file_path']}")
    print(f"Size: {report['file_size_mb']} MB")
    print(f"Within Limit: {'✓' if report['within_size_limit'] else '✗'}")
    print(f"PDF Type: {report.get('pdf_type', 'unknown')}")
    print(f"Overall Score: {report['overall_score'].upper()}")

    if report["issues"]:
        print("\n❌ ISSUES:")
        for issue in report["issues"]:
            print(f"  - {issue}")

    if report["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

    print("=" * 60)


def create_pdf_block(pdf_path: str) -> Dict[str, Any]:
    """Create a PDF document block for Claude API.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary representing a PDF document block

    Raises:
        ValueError: If PDF exceeds size limits
    """
    if not validate_pdf_size(pdf_path):
        raise ValueError(f"PDF file exceeds 32MB limit: {pdf_path}")

    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": encode_pdf(pdf_path),
        },
    }


def create_message_with_pdf(text: str, pdf_path: str) -> List[Dict[str, Any]]:
    """Create a message content list with text and a PDF.

    Args:
        text: The text prompt/question
        pdf_path: Path to the PDF file

    Returns:
        List of content blocks (PDF + text)
    """
    return [
        create_pdf_block(pdf_path),
        {"type": "text", "text": text},
    ]


def create_message_with_pdfs(
    text: str,
    pdf_paths: List[str],
    with_labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Create a message content list with text and multiple PDFs.

    Args:
        text: The text prompt/question
        pdf_paths: List of paths to PDF files
        with_labels: Optional labels for each PDF

    Returns:
        List of content blocks (PDFs + text)
    """
    content = []

    if with_labels and len(with_labels) == len(pdf_paths):
        # Add labeled PDFs
        for pdf_path, label in zip(pdf_paths, with_labels):
            content.append({"type": "text", "text": label})
            content.append(create_pdf_block(pdf_path))
    else:
        # Add all PDFs first
        for pdf_path in pdf_paths:
            content.append(create_pdf_block(pdf_path))

    # Add the main prompt at the end
    content.append({"type": "text", "text": text})

    return content


def analyze_pdf(pdf_path: str, prompt: str) -> str:
    """Analyze a PDF with a prompt.

    Args:
        pdf_path: Path to the PDF file
        prompt: Question or instruction about the PDF

    Returns:
        Claude's response text
    """
    messages = []
    content = create_message_with_pdf(prompt, pdf_path)
    add_user_message(messages, content)

    response = chat(messages)
    return text_from_message(response)


def analyze_pdfs(
    pdf_paths: List[str],
    prompt: str,
    with_labels: Optional[List[str]] = None
) -> str:
    """Analyze multiple PDFs with a prompt.

    Args:
        pdf_paths: List of paths to PDF files
        prompt: Question or instruction about the PDFs
        with_labels: Optional labels for each PDF

    Returns:
        Claude's response text
    """
    messages = []
    content = create_message_with_pdfs(prompt, pdf_paths, with_labels)
    add_user_message(messages, content)

    response = chat(messages)
    return text_from_message(response)


# ============================================
# Common PDF Analysis Tasks
# ============================================

def summarize_pdf(pdf_path: str, summary_type: str = "brief") -> str:
    """Summarize a PDF document.

    Args:
        pdf_path: Path to the PDF file
        summary_type: "brief" (1 sentence), "paragraph" (1 paragraph), or "detailed" (multiple paragraphs)

    Returns:
        PDF summary
    """
    prompts = {
        "brief": "Summarize this document in one sentence.",
        "paragraph": "Provide a one-paragraph summary of this document covering the main points.",
        "detailed": "Provide a detailed summary of this document, including main topics, key findings, and important details. Use multiple paragraphs if needed.",
    }

    prompt = prompts.get(summary_type, prompts["paragraph"])
    return analyze_pdf(pdf_path, prompt)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text content from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    prompt = "Extract all text content from this PDF document. Maintain the original structure and formatting as much as possible."
    return analyze_pdf(pdf_path, prompt)


def extract_tables_from_pdf(pdf_path: str, format_as: str = "markdown") -> str:
    """Extract tables from a PDF.

    Args:
        pdf_path: Path to the PDF file
        format_as: Output format - "markdown", "json", or "csv"

    Returns:
        Extracted tables in specified format
    """
    format_instructions = {
        "markdown": "Extract all tables from this PDF and format them as markdown tables.",
        "json": "Extract all tables from this PDF and format them as JSON arrays of objects.",
        "csv": "Extract all tables from this PDF and format them as CSV data.",
    }

    prompt = format_instructions.get(format_as, format_instructions["markdown"])
    return analyze_pdf(pdf_path, prompt)


def extract_key_information(pdf_path: str, info_type: str) -> str:
    """Extract specific types of information from a PDF.

    Args:
        pdf_path: Path to the PDF file
        info_type: Type of information - "contact", "dates", "numbers", "names", "citations"

    Returns:
        Extracted information
    """
    prompts = {
        "contact": "Extract all contact information (emails, phone numbers, addresses) from this PDF.",
        "dates": "Extract all dates mentioned in this PDF.",
        "numbers": "Extract all important numbers (statistics, measurements, amounts) from this PDF.",
        "names": "Extract all names of people, organizations, and places mentioned in this PDF.",
        "citations": "Extract all citations and references from this PDF.",
    }

    prompt = prompts.get(info_type, "Extract key information from this PDF.")
    return analyze_pdf(pdf_path, prompt)


def answer_question_about_pdf(pdf_path: str, question: str) -> str:
    """Answer a specific question about a PDF.

    Args:
        pdf_path: Path to the PDF file
        question: Specific question to answer

    Returns:
        Answer to the question
    """
    return analyze_pdf(pdf_path, question)


def compare_pdfs(pdf_path1: str, pdf_path2: str, comparison_prompt: str = None) -> str:
    """Compare two PDF documents.

    Args:
        pdf_path1: Path to first PDF
        pdf_path2: Path to second PDF
        comparison_prompt: Optional specific comparison prompt

    Returns:
        Comparison analysis
    """
    if comparison_prompt is None:
        comparison_prompt = "Compare these two documents. What are the main similarities and differences?"

    return analyze_pdfs(
        [pdf_path1, pdf_path2],
        comparison_prompt,
        with_labels=["Document 1:", "Document 2:"]
    )


def analyze_document_structure(pdf_path: str) -> str:
    """Analyze the structure and organization of a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Structure analysis
    """
    prompt = """Analyze the structure and organization of this document:
1. Document type (report, article, manual, etc.)
2. Main sections and headings
3. Number of pages (if visible)
4. Formatting elements (tables, figures, lists, etc.)
5. Overall organization and flow"""

    return analyze_pdf(pdf_path, prompt)


def extract_images_description(pdf_path: str) -> str:
    """Get descriptions of images and charts in a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Descriptions of visual elements
    """
    prompt = "Describe all images, charts, diagrams, and visual elements in this PDF. For each visual element, provide a description and explain what it shows."
    return analyze_pdf(pdf_path, prompt)


def validate_pdf_content(pdf_path: str, validation_criteria: str) -> str:
    """Validate PDF content against specific criteria.

    Args:
        pdf_path: Path to the PDF file
        validation_criteria: What to validate for

    Returns:
        Validation results
    """
    prompt = f"""Review this PDF and validate it for: {validation_criteria}

Provide:
1. Whether it meets the criteria (Yes/No)
2. Specific findings
3. Any issues or concerns
4. Recommendations if applicable"""

    return analyze_pdf(pdf_path, prompt)


def extract_structured_data(pdf_path: str, schema: str) -> str:
    """Extract data from PDF according to a specific schema.

    Args:
        pdf_path: Path to the PDF file
        schema: Description or JSON schema of data to extract

    Returns:
        Extracted structured data
    """
    prompt = f"""Extract data from this PDF according to the following schema:
{schema}

Provide the extracted data in JSON format following the schema."""

    return analyze_pdf(pdf_path, prompt)


# ============================================
# Advanced PDF Workflows
# ============================================

def create_pdf_report(pdf_path: str, report_sections: List[str]) -> Dict[str, str]:
    """Generate a comprehensive report about a PDF with multiple sections.

    Args:
        pdf_path: Path to the PDF file
        report_sections: List of report sections to generate

    Returns:
        Dictionary with section names as keys and content as values
    """
    report = {}

    for section in report_sections:
        prompt = f"Analyze this PDF and provide information about: {section}"
        report[section] = analyze_pdf(pdf_path, prompt)

    return report


def batch_process_pdfs(
    pdf_paths: List[str],
    processing_function: callable,
    **kwargs
) -> List[Dict[str, Any]]:
    """Process multiple PDFs with the same operation.

    Args:
        pdf_paths: List of PDF file paths
        processing_function: Function to apply to each PDF
        **kwargs: Additional arguments for the processing function

    Returns:
        List of results for each PDF
    """
    results = []

    for pdf_path in pdf_paths:
        try:
            result = processing_function(pdf_path, **kwargs)
            results.append({
                "pdf": pdf_path,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "pdf": pdf_path,
                "success": False,
                "error": str(e)
            })

    return results


def analyze_pdf_with_context(pdf_path: str, context: str, question: str) -> str:
    """Analyze a PDF with additional context information.

    Args:
        pdf_path: Path to the PDF file
        context: Additional context about the document
        question: Question to answer

    Returns:
        Answer based on PDF and context
    """
    prompt = f"""Context: {context}

Question: {question}

Please answer the question based on both the provided context and the content of the PDF document."""

    return analyze_pdf(pdf_path, prompt)


# ============================================
# Multi-Turn PDF Conversations
# ============================================

def pdf_conversation(pdf_path: str, initial_prompt: str) -> Dict[str, Any]:
    """Start a multi-turn conversation about a PDF.

    Args:
        pdf_path: Path to the PDF file
        initial_prompt: First question or instruction

    Returns:
        Dictionary containing messages list and first response
    """
    messages = []
    content = create_message_with_pdf(initial_prompt, pdf_path)
    add_user_message(messages, content)

    response = chat(messages)
    add_assistant_message(messages, response)

    return {
        "messages": messages,
        "response": text_from_message(response),
    }


def continue_pdf_conversation(
    conversation_state: Dict[str, Any],
    follow_up_prompt: str
) -> Dict[str, Any]:
    """Continue a conversation about a PDF.

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
# PDF + Other Content
# ============================================

def analyze_pdf_with_additional_text(
    pdf_path: str,
    additional_text: str,
    prompt: str
) -> str:
    """Analyze a PDF along with additional text content.

    Args:
        pdf_path: Path to the PDF file
        additional_text: Additional text to consider
        prompt: Analysis prompt

    Returns:
        Analysis result
    """
    messages = []
    content = [
        {"type": "text", "text": f"Additional information:\n{additional_text}\n\n"},
        create_pdf_block(pdf_path),
        {"type": "text", "text": prompt},
    ]
    add_user_message(messages, content)

    response = chat(messages)
    return text_from_message(response)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("PDF Analysis Examples")
    print("=" * 60)
    print("""
NOTE: These examples require actual PDF files to run.
Replace the placeholder paths with your own PDFs.

Maximum PDF size: 32MB
Claude can extract:
- Text content throughout the document
- Images and charts embedded in the PDF
- Tables and their data relationships
- Document structure and formatting

IMPORTANT: Install PyPDF2 for PDF type detection:
pip install PyPDF2
""")

    print("\n" + "=" * 60)
    print("Example 0: Check PDF Quality (Recommended First Step)")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document.pdf"

    # Quick quality check - tells you if it's text-based or scanned
    print_pdf_quality_report(pdf_path)

    # Or get the report as a dictionary
    report = analyze_pdf_quality(pdf_path)
    if report["overall_score"] == "good":
        print("✓ PDF is ready for Claude!")
    elif report["overall_score"] == "acceptable":
        print("⚠ PDF will work but may have some issues")
    else:
        print("✗ PDF needs preprocessing before sending to Claude")

    # Detect PDF type specifically
    type_info = detect_pdf_type(pdf_path)
    print(f"PDF Type: {type_info['type']}")
    print(f"Recommendation: {type_info['recommendation']}")
    """

    print("\n" + "=" * 60)
    print("Example 1: Summarize a PDF")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document.pdf"

    # Brief summary (1 sentence)
    brief = summarize_pdf(pdf_path, summary_type="brief")
    print("Brief Summary:")
    print(brief)

    # Detailed summary
    detailed = summarize_pdf(pdf_path, summary_type="detailed")
    print("\nDetailed Summary:")
    print(detailed)
    """

    print("\n" + "=" * 60)
    print("Example 2: Extract Text from PDF")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document.pdf"

    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:")
    print(text[:500] + "...")  # Show first 500 characters
    """

    print("\n" + "=" * 60)
    print("Example 3: Extract Tables from PDF")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/report_with_tables.pdf"

    # As markdown
    tables_md = extract_tables_from_pdf(pdf_path, format_as="markdown")
    print("Tables (Markdown):")
    print(tables_md)

    # As JSON
    tables_json = extract_tables_from_pdf(pdf_path, format_as="json")
    print("\nTables (JSON):")
    print(tables_json)
    """

    print("\n" + "=" * 60)
    print("Example 4: Extract Specific Information")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document.pdf"

    # Extract dates
    dates = extract_key_information(pdf_path, info_type="dates")
    print("Dates found:")
    print(dates)

    # Extract contact info
    contacts = extract_key_information(pdf_path, info_type="contact")
    print("\nContact information:")
    print(contacts)
    """

    print("\n" + "=" * 60)
    print("Example 5: Answer Questions About PDF")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/research_paper.pdf"

    answer1 = answer_question_about_pdf(
        pdf_path,
        "What is the main conclusion of this document?"
    )
    print("Answer:")
    print(answer1)

    answer2 = answer_question_about_pdf(
        pdf_path,
        "What methodology was used?"
    )
    print("\nMethodology:")
    print(answer2)
    """

    print("\n" + "=" * 60)
    print("Example 6: Compare Two PDFs")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF paths
    pdf1 = "path/to/version1.pdf"
    pdf2 = "path/to/version2.pdf"

    comparison = compare_pdfs(
        pdf1,
        pdf2,
        "What are the key differences between these two versions of the document?"
    )
    print("Comparison:")
    print(comparison)
    """

    print("\n" + "=" * 60)
    print("Example 7: Analyze Document Structure")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/report.pdf"

    structure = analyze_document_structure(pdf_path)
    print("Document Structure:")
    print(structure)
    """

    print("\n" + "=" * 60)
    print("Example 8: Extract Images and Charts Description")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document_with_visuals.pdf"

    visuals = extract_images_description(pdf_path)
    print("Visual Elements:")
    print(visuals)
    """

    print("\n" + "=" * 60)
    print("Example 9: Multi-Turn Conversation About PDF")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/complex_document.pdf"

    # Start conversation
    conv = pdf_conversation(
        pdf_path,
        "What is this document about?"
    )
    print("Turn 1:", conv["response"])

    # Follow-up question
    conv = continue_pdf_conversation(
        conv,
        "Can you list the main sections?"
    )
    print("\nTurn 2:", conv["response"])

    # Another follow-up
    conv = continue_pdf_conversation(
        conv,
        "What are the key findings in the conclusion section?"
    )
    print("\nTurn 3:", conv["response"])
    """

    print("\n" + "=" * 60)
    print("Example 10: Extract Structured Data")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    invoice_pdf = "path/to/invoice.pdf"

    schema = '''
    {
        "invoice_number": "string",
        "date": "string",
        "vendor": "string",
        "total_amount": "number",
        "items": [
            {
                "description": "string",
                "quantity": "number",
                "price": "number"
            }
        ]
    }
    '''

    extracted_data = extract_structured_data(invoice_pdf, schema)
    print("Extracted Invoice Data:")
    print(extracted_data)
    """

    print("\n" + "=" * 60)
    print("Example 11: Batch Process Multiple PDFs")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF paths
    pdfs = [
        "path/to/doc1.pdf",
        "path/to/doc2.pdf",
        "path/to/doc3.pdf",
    ]

    results = batch_process_pdfs(
        pdfs,
        summarize_pdf,
        summary_type="brief"
    )

    for result in results:
        if result["success"]:
            print(f"{result['pdf']}: {result['result']}")
        else:
            print(f"{result['pdf']}: ERROR - {result['error']}")
    """

    print("\n" + "=" * 60)
    print("Example 12: Validate PDF Content")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    contract_pdf = "path/to/contract.pdf"

    validation = validate_pdf_content(
        contract_pdf,
        "completeness, presence of signatures, all required sections filled out"
    )
    print("Validation Results:")
    print(validation)
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
PDF Analysis with Claude:

PDF CAPABILITIES:
Claude can extract and analyze:
✓ Text content throughout the document
✓ Images and charts embedded in the PDF
✓ Tables and their data relationships
✓ Document structure and formatting
✓ Multi-page documents (all pages analyzed together)

SUPPORTED PDF FEATURES:
- Text extraction (including OCR for scanned PDFs)
- Table extraction with structure preservation
- Image and chart descriptions
- Form data extraction
- Multiple pages (no page limit within size constraints)

PDF SIZE LIMITS:
- Maximum file size: 32MB per PDF
- No explicit page count limit
- All pages processed together (not page-by-page)
- Larger files = more tokens = higher cost

ENCODING REQUIREMENTS:
- PDFs must be base64 encoded
- Media type: "application/pdf"
- Type: "document" (not "image")
- Use standard_b64encode

MESSAGE STRUCTURE:

Single PDF:
[
    {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "<base64_string>"
        }
    },
    {"type": "text", "text": "Your prompt"}
]

Multiple PDFs:
[
    {"type": "text", "text": "Document 1:"},
    {"type": "document", "source": {...}},
    {"type": "text", "text": "Document 2:"},
    {"type": "document", "source": {...}},
    {"type": "text", "text": "Main prompt"}
]

PDF TYPE DETECTION (Human Problem vs Claude Problem):

TEXT-BASED PDFs (BEST - Claude Problem is Unlikely):
✓ Created directly from applications (Word, Google Docs, LaTeX)
✓ Text is selectable/highlightable in PDF viewer
✓ Clean text extraction
✓ Lower token costs
✓ Optimal Claude performance
✓ Example: Export directly from Word using "Save As PDF"

IMAGE-BASED/SCANNED PDFs (ACCEPTABLE - May Be Human Problem):
⚠ Scanned documents or photos of documents
⚠ Text is NOT selectable (it's part of an image)
⚠ Claude uses vision capabilities (like analyzing an image)
⚠ Higher token costs
⚠ May miss details or have extraction errors
⚠ Solution: Run through OCR software first for better accuracy

MIXED PDFs (VARIABLE):
⚠ Contains both text-based and scanned pages
⚠ Results vary by page
⚠ Claude handles it, but some pages may be more accurate than others

How to Check Your PDF:
1. Open PDF in any viewer
2. Try to select/highlight text
3. If you CAN select text → Text-based (Good!)
4. If you CANNOT select text → Image-based (Consider OCR preprocessing)

Best Export Settings from Word:
✓ File → Save As → PDF (standard format)
✓ Use "Standard" quality (not "Minimum size")
✓ Enable "Document structure tags for accessibility"
✓ Keep fonts embedded
✗ Avoid "Minimum size" compression
✗ Don't convert to images first

Using Detection Functions:
# Check PDF quality before sending
report = analyze_pdf_quality("path/to/file.pdf")
if report["overall_score"] == "poor":
    print("Fix these issues first:", report["issues"])

# Detect PDF type
type_info = detect_pdf_type("path/to/file.pdf")
if type_info["type"] == "image-based":
    print("Consider OCR preprocessing for better results")

COMMON USE CASES:

1. Document Summarization:
   - Executive summaries
   - Key points extraction
   - Abstract generation

2. Information Extraction:
   - Contact details
   - Dates and deadlines
   - Numbers and statistics
   - Names and entities

3. Table Extraction:
   - Financial data
   - Survey results
   - Comparison tables
   - Structured data

4. Document Comparison:
   - Version differences
   - Contract changes
   - Report comparisons

5. Q&A Over Documents:
   - Research papers
   - Technical manuals
   - Legal documents
   - Business reports

6. Data Validation:
   - Completeness checks
   - Compliance verification
   - Quality assurance

BEST PRACTICES:

Prompt Engineering:
✓ Be specific about what to extract
✓ Specify output format (JSON, markdown, CSV)
✓ Request structured data when applicable
✓ Use multi-turn for complex analysis
✓ Provide context when helpful

PDF Quality:
✓ Use text-based PDFs when possible (not scans)
✓ OCR scanned PDFs before sending if accuracy critical
✓ Ensure text is selectable in the PDF
✓ Optimize file size without losing quality
✓ Remove unnecessary pages

Performance:
✓ Send PDF once, ask multiple questions in conversation
✓ Use appropriate model (Sonnet for most tasks)
✓ Cache results for repeated queries
✓ Batch similar operations

Error Handling:
✓ Validate file exists before encoding
✓ Check file size before sending
✓ Handle encoding errors gracefully
✓ Verify PDF is not corrupted

COST CONSIDERATIONS:

Token Usage:
- PDFs converted to tokens for pricing
- Token count based on content, not file size
- Text-heavy PDFs = more tokens
- Images/charts in PDF also count as tokens
- Multi-page PDFs multiply token costs

Cost Optimization:
1. Remove unnecessary pages before sending
2. Use appropriate model tier
3. Cache extracted data
4. Batch similar operations
5. Send PDF once, reuse in conversation

Approximate Token Costs:
- Small document (5 pages): ~2,000-5,000 tokens
- Medium document (20 pages): ~10,000-20,000 tokens
- Large document (100 pages): ~50,000-100,000 tokens
- Varies greatly by content density

LIMITATIONS:

What Claude CAN Do:
✓ Extract text from all pages
✓ Describe images and charts within PDF
✓ Parse table structures
✓ Understand document layout
✓ Handle multi-page documents
✓ Answer questions about content
✓ Extract structured data

What Claude CANNOT Do:
✗ Edit or modify PDFs
✗ Generate new PDFs
✗ Perfect OCR on low-quality scans
✗ Process password-protected PDFs
✗ Access external links in PDFs
✗ Preserve exact formatting in output
✗ Process embedded videos or audio

MULTI-TURN CONVERSATIONS:

Pattern:
1. Send PDF in first message
2. Claude analyzes and responds
3. Ask follow-up questions
4. PDF context maintained throughout
5. No need to resend PDF

Benefits:
- Saves tokens (PDF sent once)
- Deep dive into specific sections
- Progressive analysis
- Context-aware answers

PRODUCTION TIPS:

1. Preprocessing:
   - Validate PDF integrity
   - Check file size limits
   - Remove sensitive information
   - Optimize for size

2. User Experience:
   - Show upload progress
   - Estimate processing time
   - Display page count
   - Allow page selection

3. Security:
   - Validate file type
   - Scan for malware
   - Check for sensitive data
   - Implement access controls

4. Monitoring:
   - Track token usage per PDF
   - Monitor extraction accuracy
   - Log errors and retries
   - Measure processing time

5. Caching:
   - Cache extracted data
   - Store summaries
   - Reuse analyses
   - Implement deduplication

INTEGRATION PATTERNS:

Pattern 1: Document Pipeline
PDF → Claude (extract) → Validate → Database
def process_document(pdf_path):
    summary = summarize_pdf(pdf_path)
    data = extract_structured_data(pdf_path, schema)
    validate_and_store(data)

Pattern 2: Batch Processing
for pdf in pdfs:
    result = analyze_pdf(pdf, prompt)
    results.append(result)

Pattern 3: Interactive Analysis
conversation = pdf_conversation(pdf, initial_prompt)
while user_wants_more:
    conversation = continue_pdf_conversation(conversation, next_question)

Pattern 4: Comparison Workflow
differences = compare_pdfs(old_version, new_version)
report = generate_change_report(differences)

ADVANCED TECHNIQUES:

1. Chunked Processing:
   For very large PDFs, extract specific sections:
   "Extract only the Executive Summary section"

2. Structured Output:
   Request specific formats:
   "Return as JSON with schema: {...}"

3. Multi-Document Analysis:
   Analyze relationships across documents:
   "Compare themes across these 3 reports"

4. Validation Workflows:
   "Check if this invoice matches this purchase order"

5. Data Enrichment:
   Combine PDF data with external context:
   analyze_pdf_with_context(pdf, external_data, question)

QUALITY ASSURANCE:

Best Practices:
1. Test with representative PDFs
2. Validate extraction accuracy
3. Handle edge cases (empty pages, scanned text)
4. Implement retry logic
5. Provide fallback options

Common Issues:
- Low-quality scans → Pre-process with OCR
- Large files → Split or compress
- Complex tables → Request specific format
- Mixed languages → Specify language in prompt
- Corrupted PDFs → Validate before sending

MODEL SELECTION:

Sonnet 3.5 (Recommended):
- Best balance of speed and quality
- Handles complex PDFs well
- Cost-effective for production

Opus 3:
- Highest accuracy
- Best for complex analysis
- Higher cost, slower

Haiku 3:
- Fast processing
- Good for simple extraction
- Most cost-effective
- May miss nuances

FUTURE CONSIDERATIONS:

Potential Enhancements:
- Better OCR for scanned documents
- Improved table parsing
- Page-level analysis
- Annotation support
- Form field extraction

Stay Updated:
- Monitor Anthropic docs for updates
- Test new model versions
- Optimize based on usage patterns
- Adapt to API changes
""")
