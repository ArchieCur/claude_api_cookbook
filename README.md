# Claude API Cookbook

A comprehensive collection of Python examples, patterns, and implementations for building applications with the Anthropic Claude API. This repository serves as both a learning resource and a reference guide for developers working with Claude.

## Overview

This cookbook covers the full spectrum of Claude API capabilities, from basic prompting to advanced agent architectures. Each file is a standalone example with detailed documentation, best practices, and production-ready patterns.

## Contents

### Core Capabilities

- **`Prompting.py`** - Prompt evaluation framework with automated dataset generation, testing, and grading
- **`Tool_use.py`** - Tool integration patterns including datetime tools, reminders, and batch operations
- **`Tool_streaming.py`** - Real-time streaming with tool use
- **`Tools_multi-turn.py`** - Multi-turn conversations with tool execution
- **`Structured_data.py`** - Extracting structured data from Claude responses
- **`StructuredData_Exercise.py`** - Practical exercises for structured data extraction
- **`Thinking.py`** - Extended thinking capabilities for deeper reasoning tasks
- **`Images.py`** - Image analysis, OCR, multi-image comparison, and vision capabilities
- **`PDF.py`** - PDF processing and analysis
- **`Citations.py`** - Grounding responses with citations and references

### Managed Agents

A new execution model: persistent, session-based agents running in sandboxed cloud environments. Each file is self-contained and demonstrates one capability of the Managed Agents API (research preview).

- **`managed_agents/README.md`** - Concept map, decision guide, and `client.beta.*` namespace reference
- **`managed_agents/01_multi_agent_coordinator.py`** - Coordinator + sub-agent roster, parallelization, specialization, and escalation patterns
- **`managed_agents/02_outcomes_with_rubric.py`** - Outcome-driven sessions: define a rubric, let the agent iterate until it satisfies it
- **`managed_agents/03_dreams_memory_consolidation.py`** - Async memory consolidation: deduplicate and resolve contradictions across sessions
- **`managed_agents/04_webhooks.py`** - Flask webhook handler with signature verification and idempotency
- **`managed_agents/05_advisor_strategy.py`** - Tiered Opus / Sonnet / Haiku model routing for cost-efficient multi-agent systems

### Advanced Features

- **`RAG_system.py`** - Complete RAG (Retrieval-Augmented Generation) implementation with:
  - Vector and BM25 hybrid search
  - Contextual enrichment
  - Reranking strategies
  - Multiple chunking approaches

- **`Web_search.py`** - Web search integration and information retrieval
- **`Caching.py`** - Prompt caching for cost optimization and performance
- **`Code_execution.py`** - Safe code execution capabilities
- **`MCP.py`** - Model Context Protocol implementation guide for building servers and clients
- **`Agents_and_Workflows.py`** - Architectural patterns:
  - Workflows vs Agents decision framework
  - Parallelization patterns
  - Chaining strategies
  - Routing workflows

### Evaluation & Testing

- **`Running_Eval.py`** - Evaluation framework for testing prompts
- **`Test_Datasets.py`** - Dataset generation and management
- **`Prompt_Dataset.py`** - Prompt testing datasets
- **`Code_Grading.py`** - Automated code evaluation
- **`Grader_Exercise.py`** - Grading system exercises
- **`Model_Grading.py`** - Model output evaluation

### Utilities & Demos

- **`streaming_demo.py`** - Streaming response demonstrations
- **`prefill_and_stop_demo.py`** - Prefill and stop sequence examples
- **`stop_sequences_template.py`** - Stop sequence patterns
- **`structured_data_template.py`** - Templates for structured output
- **`test_client.py`** - Client testing utilities
- **`Text_editor_tool.py`** - Text editing tool implementation
- **`Claude_Code.py`** - Claude Code integration examples

## Prerequisites

```bash
pip install anthropic python-dotenv voyageai
```

For Managed Agents webhook examples:
```bash
pip install flask
```

For MCP examples:
```bash
pip install "mcp[cli]>=1.8.0"
```

## Setup

1. Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your_api_key_here
VOYAGE_API_KEY=your_voyage_key_here  # For RAG examples
```

2. Each Python file is self-contained and can be run independently:
```bash
python Prompting.py
python Tool_use.py
python RAG_system.py
```

## Key Features

### 🎯 Production-Ready Patterns
Each example includes error handling, best practices, and considerations for production deployment.

### 📚 Comprehensive Documentation
Inline documentation explains not just how to use features, but when and why to use them.

### 🔧 Modular Design
Helper functions and reusable components make it easy to integrate patterns into your own projects.

### 🚀 Progressive Learning
Examples progress from basic concepts to advanced architectures, building on earlier patterns.

## Architecture Patterns

### Workflows
- **Parallelization** - Split complex evaluations into focused, parallel tasks
- **Chaining** - Sequential processing for multi-step refinement
- **Routing** - Direct requests to specialized pipelines

### Agents
- **Tool Design** - Abstract, composable tools for maximum flexibility
- **Environment Inspection** - Verify actions and adapt behavior
- **Multi-Turn Planning** - Complex task decomposition

## Use Cases

This cookbook demonstrates implementations for:

- 💬 Conversational AI applications
- 📊 Data extraction and analysis
- 🔍 Semantic search and RAG systems
- 🤖 Autonomous agents and workflows
- 📝 Document processing and OCR
- 🧮 Complex reasoning and problem-solving
- 🔧 Tool integration and API orchestration

## Claude's Contribution

This repository was developed with significant contributions from Claude Code (Anthropic) in collaboration with ArchieCur.

**Original cookbook** (2025): powered by Claude Sonnet 4.5 (claude-sonnet-4-5-20250929). Claude Code assisted in:

- **Architecture Design** - Designing modular, extensible patterns for each capability
- **Code Implementation** - Writing production-ready Python code with proper error handling and type hints
- **Documentation** - Creating comprehensive inline documentation and pattern explanations
- **Best Practices** - Incorporating lessons learned from real-world Claude API usage
- **Educational Structure** - Organizing examples progressively from basic to advanced concepts
- **Testing & Validation** - Ensuring examples are practical and production-ready

**Managed Agents update** (May 2026): powered by Claude Sonnet 4.6 (claude-sonnet-4-6). Added the `managed_agents/` folder covering the Managed Agents API research preview — six self-contained files spanning multi-agent coordination, outcome-driven sessions, Dreams memory consolidation, webhooks, and the advisor strategy pattern.

The collaboration between human guidance and Claude's technical capabilities resulted in a resource that bridges theoretical understanding with practical implementation.

## Best Practices

### 🔒 Security
- Never commit API keys (use `.env` files)
- Validate and sanitize user inputs
- Implement rate limiting for production use

### 💰 Cost Optimization
- Use prompt caching for repeated patterns
- Choose appropriate models (Haiku for simple tasks, Sonnet for balance, Opus for quality)
- Batch similar requests when possible
- Monitor token usage

### ⚡ Performance
- Implement streaming for better UX
- Use async/await for concurrent operations
- Cache embeddings and frequent computations
- Optimize image sizes before sending

### 🎯 Reliability
- Start with workflows before resorting to agents
- Implement proper error handling and retries
- Test prompts systematically with evaluation frameworks
- Monitor and log for continuous improvement

## Examples in Action

### Simple Tool Use
```python
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=[get_current_datetime_schema],
    messages=[{"role": "user", "content": "What time is it?"}]
)
```

### RAG System
```python
from RAG_system import RAGSystem

rag = RAGSystem(
    chunking_strategy="sentences",
    use_contextual_enrichment=True,
    use_reranking=True
)

rag.add_documents(documents)
answer = rag.query("Your question here", k=3)
```

### Extended Thinking
```python
response = chat(
    messages,
    thinking=True,
    thinking_budget=2048
)
```

## Contributing

This repository is designed as a learning resource. Feel free to:
- Use these patterns in your own projects
- Adapt examples to your specific needs
- Share feedback or suggestions
- Report issues or inconsistencies

## Resources

- [Anthropic Documentation](https://docs.anthropic.com/)
- [Claude API Reference](https://docs.anthropic.com/en/api/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## License

This is an educational resource. Use these examples as reference for your own projects.

---

**Built with Claude** - Demonstrating the power of human-AI collaboration in creating comprehensive technical resources.
