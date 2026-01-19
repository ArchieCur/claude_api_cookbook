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
model = "claude-sonnet-4-5"  # Citations require Sonnet 3.5+ or Opus 3+

"""
IMPORTANT: Citations Limitations

Citations work for TEXTUAL DOCUMENT CONTENT ONLY:
✓ Raw text (text/plain)
✓ PDFs (application/pdf)
✓ Markdown, HTML (as text)

Citations DO NOT work for:
✗ Images (use "image" type, not "document")
✗ Videos or audio
✗ Web pages (must fetch and convert to text first)

If you want citations from web pages, images with text, or other sources,
you need to convert them to text first before sending to Claude.
"""


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
    """Make an API call to Claude with optional citations."""
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
# Document Creation with Citations
# ============================================

def create_text_document_with_citations(
    text: str,
    title: str = "Document",
    enable_citations: bool = True
) -> Dict[str, Any]:
    """Create a text document block with citations enabled.

    Args:
        text: The text content of the document
        title: Document title/identifier
        enable_citations: Whether to enable citations (default: True)

    Returns:
        Dictionary representing a text document block with citations
    """
    return {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": text,
        },
        "title": title,
        "citations": {"enabled": enable_citations},
    }


def create_pdf_document_with_citations(
    pdf_path: str,
    title: Optional[str] = None,
    enable_citations: bool = True
) -> Dict[str, Any]:
    """Create a PDF document block with citations enabled.

    Args:
        pdf_path: Path to the PDF file
        title: Document title/identifier (defaults to filename)
        enable_citations: Whether to enable citations (default: True)

    Returns:
        Dictionary representing a PDF document block with citations
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Use filename as title if not provided
    if title is None:
        title = Path(pdf_path).stem

    with open(pdf_path, "rb") as pdf_file:
        pdf_base64 = base64.standard_b64encode(pdf_file.read()).decode("utf-8")

    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf_base64,
        },
        "title": title,
        "citations": {"enabled": enable_citations},
    }


def create_message_with_document(
    document_block: Dict[str, Any],
    question: str
) -> List[Dict[str, Any]]:
    """Create a message with a document and a question.

    Args:
        document_block: Document block (from create_text_document_with_citations or create_pdf_document_with_citations)
        question: Question to ask about the document

    Returns:
        List of content blocks for the message
    """
    return [
        document_block,
        {"type": "text", "text": question},
    ]


def create_message_with_multiple_documents(
    document_blocks: List[Dict[str, Any]],
    question: str
) -> List[Dict[str, Any]]:
    """Create a message with multiple documents and a question.

    Args:
        document_blocks: List of document blocks
        question: Question to ask about the documents

    Returns:
        List of content blocks for the message
    """
    content = document_blocks.copy()
    content.append({"type": "text", "text": question})
    return content


# ============================================
# Citation Extraction
# ============================================

def extract_citations(message) -> List[Dict[str, Any]]:
    """Extract all citations from a message response.

    Args:
        message: Message object from Claude

    Returns:
        List of citations with their details
    """
    citations = []

    for block in message.content:
        if block.type == "text" and hasattr(block, 'citations'):
            for citation in block.citations:
                citations.append({
                    "cited_text": citation.cited_text,
                    "document_title": citation.document_title,
                    "start_char": getattr(citation, 'start_char', None),
                    "end_char": getattr(citation, 'end_char', None),
                })

    return citations


def has_citations(message) -> bool:
    """Check if a message contains any citations.

    Args:
        message: Message object from Claude

    Returns:
        True if message contains citations
    """
    for block in message.content:
        if block.type == "text" and hasattr(block, 'citations'):
            if len(block.citations) > 0:
                return True
    return False


def format_response_with_citations(message) -> str:
    """Format a response with inline citation markers.

    Args:
        message: Message object from Claude

    Returns:
        Formatted text with citation markers
    """
    formatted_text = ""

    for block in message.content:
        if block.type == "text":
            text = block.text
            formatted_text += text

            if hasattr(block, 'citations') and len(block.citations) > 0:
                formatted_text += "\n\n--- CITATIONS ---\n"
                for i, citation in enumerate(block.citations, 1):
                    formatted_text += f"[{i}] From '{citation.document_title}': \"{citation.cited_text}\"\n"

    return formatted_text


# ============================================
# Q&A with Citations
# ============================================

def ask_with_citations(
    text: str,
    question: str,
    title: str = "Document"
) -> Dict[str, Any]:
    """Ask a question about text with citation support.

    Args:
        text: The text content to analyze
        question: Question to ask
        title: Document title

    Returns:
        Dictionary with answer and citations
    """
    messages = []

    # Create document with citations
    doc_block = create_text_document_with_citations(text, title)
    content = create_message_with_document(doc_block, question)

    add_user_message(messages, content)
    response = chat(messages)

    return {
        "answer": text_from_message(response),
        "citations": extract_citations(response),
        "full_response": response,
    }


def ask_pdf_with_citations(
    pdf_path: str,
    question: str,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Ask a question about a PDF with citation support.

    Args:
        pdf_path: Path to PDF file
        question: Question to ask
        title: Document title (defaults to filename)

    Returns:
        Dictionary with answer and citations
    """
    messages = []

    # Create PDF document with citations
    doc_block = create_pdf_document_with_citations(pdf_path, title)
    content = create_message_with_document(doc_block, question)

    add_user_message(messages, content)
    response = chat(messages)

    return {
        "answer": text_from_message(response),
        "citations": extract_citations(response),
        "full_response": response,
    }


def ask_multiple_documents_with_citations(
    documents: List[Dict[str, str]],
    question: str
) -> Dict[str, Any]:
    """Ask a question across multiple documents with citations.

    Args:
        documents: List of dicts with 'text' and 'title' keys
        question: Question to ask

    Returns:
        Dictionary with answer and citations from all documents
    """
    messages = []

    # Create document blocks for each document
    doc_blocks = [
        create_text_document_with_citations(doc["text"], doc["title"])
        for doc in documents
    ]

    content = create_message_with_multiple_documents(doc_blocks, question)

    add_user_message(messages, content)
    response = chat(messages)

    return {
        "answer": text_from_message(response),
        "citations": extract_citations(response),
        "full_response": response,
    }


# ============================================
# Multi-Turn Q&A with Citations
# ============================================

def start_citation_conversation(
    text: str,
    initial_question: str,
    title: str = "Document"
) -> Dict[str, Any]:
    """Start a multi-turn conversation with citation support.

    Args:
        text: The text content
        initial_question: First question
        title: Document title

    Returns:
        Conversation state with messages and response
    """
    messages = []

    doc_block = create_text_document_with_citations(text, title)
    content = create_message_with_document(doc_block, initial_question)

    add_user_message(messages, content)
    response = chat(messages)
    add_assistant_message(messages, response)

    return {
        "messages": messages,
        "answer": text_from_message(response),
        "citations": extract_citations(response),
    }


def continue_citation_conversation(
    conversation_state: Dict[str, Any],
    follow_up_question: str
) -> Dict[str, Any]:
    """Continue a conversation with follow-up questions.

    Args:
        conversation_state: State from previous turn
        follow_up_question: Next question

    Returns:
        Updated conversation state
    """
    messages = conversation_state["messages"]
    add_user_message(messages, follow_up_question)

    response = chat(messages)
    add_assistant_message(messages, response)

    return {
        "messages": messages,
        "answer": text_from_message(response),
        "citations": extract_citations(response),
    }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Citations Examples")
    print("=" * 60)
    print("""
IMPORTANT: Citations Limitations

Citations work for TEXTUAL DOCUMENT CONTENT ONLY:
✓ Raw text (text/plain)
✓ PDFs (application/pdf)
✓ Markdown, HTML (as text)

Citations DO NOT work for:
✗ Images (use "image" type, not "document")
✗ Videos or audio
✗ Web pages (must fetch and convert to text first)

If you want citations from web pages, images with text, or other sources,
you need to convert them to text first before sending to Claude.
""")

    print("\n" + "=" * 60)
    print("Example 1: Simple Q&A with Citations")
    print("=" * 60)

    """
    # Uncomment to run
    sample_text = \"\"\"
    Python was created by Guido van Rossum and first released in 1991.
    Python 3.0 was released in 2008 and introduced many breaking changes.
    The language emphasizes code readability with significant whitespace.
    Python supports multiple programming paradigms including object-oriented,
    functional, and procedural programming.
    \"\"\"

    result = ask_with_citations(
        sample_text,
        "Who created Python and when?",
        title="Python History"
    )

    print("Answer:", result["answer"])
    print("\nCitations:")
    for i, citation in enumerate(result["citations"], 1):
        print(f"[{i}] From '{citation['document_title']}': \"{citation['cited_text']}\"")
    """

    print("\n" + "=" * 60)
    print("Example 2: PDF with Citations")
    print("=" * 60)

    """
    # Uncomment and replace with your PDF path
    pdf_path = "path/to/document.pdf"

    result = ask_pdf_with_citations(
        pdf_path,
        "What are the main findings of this report?",
        title="Research Report"
    )

    print("Answer:", result["answer"])
    print("\nCitations:")
    for citation in result["citations"]:
        print(f"- \"{citation['cited_text']}\" (from {citation['document_title']})")
    """

    print("\n" + "=" * 60)
    print("Example 3: Multiple Documents with Citations")
    print("=" * 60)

    """
    # Uncomment to run
    documents = [
        {
            "title": "Python Basics",
            "text": "Python is an interpreted, high-level programming language. "
                    "It was created by Guido van Rossum in 1991."
        },
        {
            "title": "Python Features",
            "text": "Python emphasizes code readability and uses significant whitespace. "
                    "It supports object-oriented, functional, and procedural programming."
        },
        {
            "title": "Python Versions",
            "text": "Python 3.0 was released in December 2008. "
                    "It introduced many breaking changes from Python 2.x."
        }
    ]

    result = ask_multiple_documents_with_citations(
        documents,
        "Tell me about Python's history and features"
    )

    print("Answer:", result["answer"])
    print("\nCitations:")
    for citation in result["citations"]:
        print(f"- [{citation['document_title']}] \"{citation['cited_text']}\"")
    """

    print("\n" + "=" * 60)
    print("Example 4: Multi-Turn Conversation with Citations")
    print("=" * 60)

    """
    # Uncomment to run
    article = \"\"\"
    Climate change is causing global temperatures to rise.
    The primary driver is greenhouse gas emissions from human activities.
    Scientists predict average temperatures could rise 1.5-2°C by 2050.
    This will lead to more extreme weather events and sea level rise.
    Renewable energy and carbon capture are key mitigation strategies.
    \"\"\"

    # First question
    conv = start_citation_conversation(
        article,
        "What is causing climate change?",
        title="Climate Article"
    )
    print("Turn 1:", conv["answer"])
    print("Citations:", len(conv["citations"]))

    # Follow-up question
    conv = continue_citation_conversation(
        conv,
        "What are the predicted impacts?"
    )
    print("\nTurn 2:", conv["answer"])
    print("Citations:", len(conv["citations"]))

    # Another follow-up
    conv = continue_citation_conversation(
        conv,
        "What can we do about it?"
    )
    print("\nTurn 3:", conv["answer"])
    print("Citations:", len(conv["citations"]))
    """

    print("\n" + "=" * 60)
    print("Example 5: Formatted Output with Citations")
    print("=" * 60)

    """
    # Uncomment to run
    text = "Earth's atmosphere formed about 4 billion years ago through volcanic outgassing."

    result = ask_with_citations(
        text,
        "How did Earth's atmosphere form?",
        title="Earth Science"
    )

    # Pretty print with citations
    formatted = format_response_with_citations(result["full_response"])
    print(formatted)
    """

    print("\n" + "=" * 60)
    print("Example 6: Check for Citations")
    print("=" * 60)

    """
    # Uncomment to run
    text = "The sky is blue due to Rayleigh scattering."

    result = ask_with_citations(text, "Why is the sky blue?")

    if has_citations(result["full_response"]):
        print("✓ Response includes citations")
        print(f"  Found {len(result['citations'])} citation(s)")
    else:
        print("✗ No citations in response")
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Citations with Claude:

WHAT SUPPORTS CITATIONS:

✓ Text Documents:
  {
      "type": "document",
      "source": {
          "type": "text",
          "media_type": "text/plain",
          "data": "Your text here..."
      },
      "title": "Document Title",
      "citations": {"enabled": true}
  }

✓ PDF Documents:
  {
      "type": "document",
      "source": {
          "type": "base64",
          "media_type": "application/pdf",
          "data": "<base64_pdf_data>"
      },
      "title": "PDF Title",
      "citations": {"enabled": true}
  }

✓ Text-based formats:
  - Plain text (text/plain)
  - Markdown (as text)
  - HTML (as text)
  - Any text content

WHAT DOES NOT SUPPORT CITATIONS:

✗ Images:
  - Images use type "image", not "document"
  - No citations field available
  - Claude analyzes visually, not as citable text

✗ Videos/Audio:
  - Not supported by Claude API

✗ Web Pages (directly):
  - Must fetch and convert to text first
  - See workaround below

WORKAROUND FOR WEB PAGES:

import requests
from bs4 import BeautifulSoup

# Fetch web page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()

# Create document with citations
doc = create_text_document_with_citations(
    text,
    title="Web Page Title"
)

KEY FEATURES:

1. Source Attribution:
   - Claude cites specific text from documents
   - Citations include document title
   - Character positions available (start_char, end_char)

2. Multiple Documents:
   - Send multiple documents in one request
   - Citations distinguish between sources
   - Cross-document Q&A supported

3. Multi-Turn Conversations:
   - Document sent once in first message
   - Follow-up questions maintain citation support
   - Context preserved throughout conversation

4. Citation Structure:
   {
       "cited_text": "The actual quoted text",
       "document_title": "Source document name",
       "start_char": 123,
       "end_char": 456
   }

BENEFITS:

✓ Verifiable Answers:
  - Users can verify claims in source documents
  - Reduces hallucinations
  - Builds trust in AI responses

✓ Source Transparency:
  - Clear attribution to specific documents
  - Exact quotes provided
  - Character positions for precise lookup

✓ Research Applications:
  - Academic research with proper citations
  - Legal document analysis
  - Medical literature review
  - Compliance and auditing

BEST PRACTICES:

1. Document Titles:
   - Use descriptive titles
   - Helps users understand citation sources
   - Important for multi-document scenarios

2. Text Quality:
   - Clean, well-formatted text produces better citations
   - Remove unnecessary whitespace/formatting
   - Structure matters for accurate citations

3. Question Design:
   - Ask specific questions for precise citations
   - Request evidence: "What evidence supports..."
   - Explicitly ask for citations: "With citations, explain..."

4. Verification:
   - Always check citations against source
   - Verify character positions if using them
   - Use has_citations() to confirm presence

5. Multiple Documents:
   - Use distinct, clear titles
   - Organize by topic or type
   - Keep documents focused

USE CASES:

1. Research Q&A:
   - Ask questions about papers/articles
   - Get cited answers with sources
   - Build literature reviews

2. Document Analysis:
   - Analyze contracts with citations
   - Review policies with source attribution
   - Compare documents with references

3. Fact Checking:
   - Verify claims against sources
   - Get exact quotes supporting answers
   - Cross-reference multiple documents

4. Education:
   - Study materials with citations
   - Research assistance
   - Academic writing support

5. Legal/Compliance:
   - Policy interpretation with citations
   - Regulatory analysis with sources
   - Contract review with references

LIMITATIONS:

1. Citation Accuracy:
   - Citations generally accurate but verify important claims
   - Character positions may vary with formatting
   - Edge cases with complex documents

2. Citation Granularity:
   - Cites text passages, not specific data points
   - May cite broader context than needed
   - Multiple citations for complex answers

3. Document Length:
   - Very long documents may have citation challenges
   - Consider chunking extremely large texts
   - Balance detail vs. context

4. Format Limitations:
   - Text and PDF only
   - No support for images, audio, video
   - Web pages require preprocessing

PRODUCTION TIPS:

1. Validation:
   - Implement citation verification
   - Check citation completeness
   - Validate character positions

2. User Experience:
   - Display citations as footnotes
   - Provide links to source documents
   - Highlight cited text in original

3. Error Handling:
   - Handle missing citations gracefully
   - Fallback if citations unavailable
   - Log citation failures

4. Performance:
   - Cache document embeddings if possible
   - Batch multi-document requests
   - Consider pagination for many documents

5. Storage:
   - Store documents with stable IDs
   - Index citations for retrieval
   - Link citations to original sources

CITATION FORMAT OPTIONS:

1. Inline Citations:
   "Answer text [1][2]"
   [1] Source quote from Doc A
   [2] Source quote from Doc B

2. Footnotes:
   Answer with superscript numbers
   Footnotes at bottom with full citations

3. Structured Output:
   {
       "answer": "...",
       "sources": [
           {"text": "...", "doc": "..."}
       ]
   }

4. Academic Style:
   Author-date or numbered references
   Full bibliographic information

INTEGRATION PATTERNS:

Pattern 1: Simple Q&A Service
def qa_with_sources(text, question):
    result = ask_with_citations(text, question)
    return {
        "answer": result["answer"],
        "sources": result["citations"]
    }

Pattern 2: Document Chatbot
conversation = start_citation_conversation(doc, question)
while user_has_questions:
    conversation = continue_citation_conversation(
        conversation,
        next_question
    )

Pattern 3: Multi-Document Research
documents = load_documents()
results = ask_multiple_documents_with_citations(
    documents,
    research_question
)

Pattern 4: Citation Verification
result = ask_with_citations(text, question)
for citation in result["citations"]:
    verify_citation_in_source(citation, original_text)

FUTURE ENHANCEMENTS:

- Support for more document types
- Improved citation granularity
- Citation confidence scores
- Automatic bibliography generation
- Citation export formats
""")
