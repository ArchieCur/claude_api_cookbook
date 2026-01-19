from dotenv import load_dotenv
load_dotenv()
#Voyageai is the embedding generator used in the course, replace the import with the genertor you use and add that API key to env variables. 
from anthropic import Anthropic
from anthropic.types import Message
import voyageai #replace with your embeding generator
import json
import math
import re
from collections import Counter
from typing import Optional, Any, List, Dict, Tuple, Callable, Protocol
import random
import string

# Load env variables and create clients
anthropic_client = Anthropic()
embedding_client = voyageai.Client() #replace with your embedding client API key
model = "claude-3-5-haiku-20241022"  # Fast and cost-effective for reranking/context


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


def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
    """Make an API call to Claude."""
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = anthropic_client.messages.create(**params)
    return message


def text_from_message(message):
    """Extract all text content from a message."""
    return "\n".join([block.text for block in message.content if block.type == "text"])


# ============================================
# Chunking Strategies
# ============================================

def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into fixed-size character chunks with overlap.

    Args:
        text: The text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> List[str]:
    """Split text into chunks of N sentences with overlap.

    Args:
        text: The text to chunk
        sentences_per_chunk: Number of sentences per chunk
        overlap: Number of overlapping sentences between chunks

    Returns:
        List of text chunks
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunks.append(' '.join(chunk_sentences))
        i += sentences_per_chunk - overlap

    return chunks


def chunk_by_section(text: str, section_pattern: str = r"\n## ") -> List[str]:
    """Split text by section markers (e.g., markdown headers).

    Args:
        text: The text to chunk
        section_pattern: Regex pattern for section boundaries

    Returns:
        List of text chunks (one per section)
    """
    return re.split(section_pattern, text)


def chunk_by_paragraphs(text: str, paragraphs_per_chunk: int = 2) -> List[str]:
    """Split text by paragraphs.

    Args:
        text: The text to chunk
        paragraphs_per_chunk: Number of paragraphs per chunk

    Returns:
        List of text chunks
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []

    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        chunk = '\n\n'.join(paragraphs[i:i + paragraphs_per_chunk])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================
# Embedding Generation
# ============================================

def generate_embedding(
    chunks,
    model: str = "voyage-3-large",
    input_type: str = "document"
) -> List[List[float]]:
    """Generate embeddings using Voyage AI.

    Args:
        chunks: Single string or list of strings to embed
        model: Voyage AI model to use
        input_type: "document" for corpus, "query" for search queries

    Returns:
        Single embedding or list of embeddings
    """
    is_list = isinstance(chunks, list)
    input_data = chunks if is_list else [chunks]
    result = embedding_client.embed(input_data, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]


# ============================================
# Vector Index Implementation
# ============================================

class VectorIndex:
    """Semantic search index using embeddings.

    Stores document embeddings and performs similarity search.
    Supports cosine and euclidean distance metrics.
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn: Optional[Callable] = None,
    ):
        """Initialize vector index.

        Args:
            distance_metric: "cosine" or "euclidean"
            embedding_fn: Function to generate embeddings from text
        """
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None

        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")

        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn or generate_embedding

    def add_document(self, document: Dict[str, Any]):
        """Add a single document to the index."""
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents to the index (batch operation)."""
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        contents = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain a 'content' key.")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string.")
            contents.append(doc["content"])

        vectors = self._embedding_fn(contents)

        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """Search for k most similar documents.

        Args:
            query: Query string or embedding vector
            k: Number of results to return

        Returns:
            List of (document, distance) tuples sorted by similarity
        """
        if not self.vectors:
            return []

        if isinstance(query, str):
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        dist_func = self._cosine_distance if self._distance_metric == "cosine" else self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        distances.sort(key=lambda item: item[0])
        return [(doc, dist) for dist, doc in distances[:k]]

    def add_vector(self, vector: List[float], document: Dict[str, Any]):
        """Add a pre-computed vector and document."""
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product of two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        """Calculate magnitude of a vector."""
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine distance (1 - cosine similarity)."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"


# ============================================
# BM25 Index Implementation
# ============================================

class BM25Index:
    """Lexical search index using BM25 algorithm.

    Performs keyword-based search using the BM25 ranking function.
    Complementary to semantic search for hybrid retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (0-1, typically 0.75)
            tokenizer: Function to tokenize text (defaults to simple word splitting)
        """
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase and split on non-word characters."""
        text = text.lower()
        tokens = re.split(r'\W+', text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        """Update index statistics when adding a document."""
        self._doc_len.append(len(doc_tokens))

        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

        self._index_built = False

    def _calculate_idf(self):
        """Calculate IDF (Inverse Document Frequency) for all terms."""
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        """Build the index (calculate statistics)."""
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        """Add a single document to the index."""
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)

        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents to the index (batch operation)."""
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain a 'content' key.")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string.")

            content = doc["content"]
            doc_tokens = self._tokenizer(content)

            self.documents.append(doc)
            self._corpus_tokens.append(doc_tokens)
            self._update_stats_add(doc_tokens)

        self._index_built = False

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        """Compute BM25 score for a document given query tokens."""
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self,
        query: Any,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for k most relevant documents using BM25.

        Args:
            query: Query string
            k: Number of results to return
            score_normalization_factor: Factor for exponential score normalization

        Returns:
            List of (document, normalized_distance) tuples
        """
        if not self.documents:
            return []

        if isinstance(query, str):
            query_text = query
        else:
            raise TypeError("Query must be a string for BM25Index.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        # Normalize scores to distance-like metric for compatibility with vector search
        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])
        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"


# ============================================
# Hybrid Retriever with RRF Fusion
# ============================================

class SearchIndex(Protocol):
    """Protocol defining the interface for search indexes."""
    def add_document(self, document: Dict[str, Any]) -> None: ...
    def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...
    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]: ...


class Retriever:
    """Hybrid retriever using multiple indexes with RRF fusion.

    Combines results from multiple search indexes (e.g., vector + BM25)
    using Reciprocal Rank Fusion. Optionally applies LLM-based reranking.
    """

    def __init__(
        self,
        *indexes: SearchIndex,
        reranker_fn: Optional[Callable[[List[Dict[str, Any]], str, int], List[str]]] = None,
    ):
        """Initialize retriever with one or more search indexes.

        Args:
            *indexes: One or more SearchIndex instances
            reranker_fn: Optional function for LLM-based reranking
        """
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)
        self._reranker_fn = reranker_fn

    def add_document(self, document: Dict[str, Any]):
        """Add a document to all indexes."""
        if "id" not in document:
            document["id"] = "".join(
                random.choices(string.ascii_letters + string.digits, k=8)
            )

        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents to all indexes (batch operation)."""
        # Ensure all documents have IDs
        for doc in documents:
            if "id" not in doc:
                doc["id"] = "".join(
                    random.choices(string.ascii_letters + string.digits, k=8)
                )

        for index in self._indexes:
            index.add_documents(documents)

    def search(
        self,
        query_text: str,
        k: int = 1,
        k_rrf: int = 60,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search using hybrid retrieval with RRF (Recipricol Rank Fusion).

        Args:
            query_text: Search query
            k: Number of final results to return
            k_rrf: RRF constant (typically 60)

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        # Query all indexes (retrieve more results for better fusion)
        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        # Build rank map for each document
        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        # Calculate RRF scores
        def calc_rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored_docs: List[Tuple[Dict[str, Any], float]] = [
            (ranks["doc_obj"], calc_rrf_score(ranks["ranks"]))
            for ranks in doc_ranks.values()
        ]

        # Filter and sort by RRF score
        filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        result = filtered_docs[:k]

        # Apply reranking if provided
        if self._reranker_fn is not None:
            docs_only = [doc for doc, _ in result]

            # Ensure all docs have IDs
            for doc in docs_only:
                if "id" not in doc:
                    doc["id"] = "".join(
                        random.choices(string.ascii_letters + string.digits, k=8)
                    )

            doc_lookup = {doc["id"]: doc for doc in docs_only}
            reranked_ids = self._reranker_fn(docs_only, query_text, k)

            new_result = []
            original_scores = {id(doc): score for doc, score in result}

            for doc_id in reranked_ids:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    score = original_scores.get(id(doc), 0.0)
                    new_result.append((doc, score))

            result = new_result

        return result


# ============================================
# Contextual Enrichment
# ============================================

def add_contextual_info(
    chunk: str,
    surrounding_context: str,
    use_claude: bool = True
) -> str:
    """Add contextual information to a chunk before embedding.

    Args:
        chunk: The text chunk to enrich
        surrounding_context: Context from surrounding chunks or document
        use_claude: If True, use Claude to generate context. If False, prepend raw context.

    Returns:
        Enriched chunk with context prepended
    """
    if not use_claude:
        # Simple prepend strategy
        return f"Context: {surrounding_context[:200]}...\n\n{chunk}"

    # Use Claude to generate succinct context
    prompt = f"""
Write a short and succinct snippet of text to situate this chunk within the
overall document context for the purposes of improving search retrieval.

Here is the surrounding context:
<context>
{surrounding_context}
</context>

Here is the chunk we want to situate:
<chunk>
{chunk}
</chunk>

Answer only with the succinct context (1-2 sentences) and nothing else.
"""

    messages = []
    add_user_message(messages, prompt)
    result = chat(messages)

    context_snippet = text_from_message(result)
    return context_snippet + "\n\n" + chunk


def enrich_chunks_with_context(
    chunks: List[str],
    full_document: str = "",
    num_start_chunks: int = 2,
    num_prev_chunks: int = 2,
    use_claude: bool = True
) -> List[str]:
    """Enrich all chunks with contextual information.

    Args:
        chunks: List of text chunks
        full_document: Full source document (optional)
        num_start_chunks: Number of chunks from document start to include as context
        num_prev_chunks: Number of previous chunks to include as context
        use_claude: Whether to use Claude for context generation

    Returns:
        List of enriched chunks
    """
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        context_parts = []

        # Add initial chunks (document overview)
        if num_start_chunks > 0:
            context_parts.extend(chunks[:min(num_start_chunks, len(chunks))])

        # Add previous chunks (local context)
        if num_prev_chunks > 0:
            start_idx = max(0, i - num_prev_chunks)
            context_parts.extend(chunks[start_idx:i])

        # Build context string
        if context_parts:
            context = "\n\n".join(context_parts)
        elif full_document:
            context = full_document[:1000]  # Use document beginning
        else:
            context = ""

        if context:
            enriched_chunk = add_contextual_info(chunk, context, use_claude)
        else:
            enriched_chunk = chunk

        enriched_chunks.append(enriched_chunk)

    return enriched_chunks


# ============================================
# Reranking
# ============================================

def create_reranker_fn(
    top_k: int = None,
    use_json: bool = True
) -> Callable[[List[Dict[str, Any]], str, int], List[str]]:
    """Create a reranker function using Claude.

    Args:
        top_k: Maximum number of documents to rerank (None = use all)
        use_json: Use JSON format for structured output

    Returns:
        Reranker function
    """
    def reranker_fn(docs: List[Dict[str, Any]], query_text: str, k: int) -> List[str]:
        """Rerank documents using Claude's deep understanding.

        Args:
            docs: List of documents to rerank
            query_text: User's query
            k: Number of top documents to return

        Returns:
            List of document IDs sorted by relevance
        """
        # Format documents as XML
        joined_docs = "\n".join([
            f"""<document>
<document_id>{doc["id"]}</document_id>
<document_content>{doc["content"][:500]}</document_content>
</document>"""
            for doc in docs[:top_k] if top_k else docs
        ])

        prompt = f"""You are a search relevance expert. Your task is to select and rank the {k} most relevant documents to answer the user's question.

Here is the user's question:
<question>
{query_text}
</question>

Here are the candidate documents:
<documents>
{joined_docs}
</documents>

Respond in the following format:
```json
{{
    "document_ids": ["id1", "id2", ...]
}}
```

List exactly {k} document IDs, ordered from most to least relevant."""

        messages = []
        add_user_message(messages, prompt)
        add_assistant_message(messages, "```json")

        result = chat(messages, stop_sequences=["```"])

        try:
            return json.loads(text_from_message(result))["document_ids"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: return original order
            return [doc["id"] for doc in docs[:k]]

    return reranker_fn


# ============================================
# End-to-End RAG System
# ============================================

class RAGSystem:
    """Complete RAG system with hybrid search, contextual enrichment, and reranking.

    This class orchestrates the full RAG pipeline:
    1. Document chunking
    2. Contextual enrichment (optional)
    3. Indexing (vector + BM25)
    4. Hybrid search with RRF fusion
    5. LLM reranking (optional)
    6. Response generation with Claude
    """

    def __init__(
        self,
        chunking_strategy: str = "sentences",
        use_contextual_enrichment: bool = True,
        use_reranking: bool = True,
        embedding_fn: Optional[Callable] = None,
    ):
        """Initialize RAG system.

        Args:
            chunking_strategy: "characters", "sentences", "sections", or "paragraphs"
            use_contextual_enrichment: Whether to enrich chunks with context
            use_reranking: Whether to use LLM reranking
            embedding_fn: Custom embedding function (defaults to Voyage AI)
        """
        self.chunking_strategy = chunking_strategy
        self.use_contextual_enrichment = use_contextual_enrichment
        self.use_reranking = use_reranking

        # Initialize indexes
        self.vector_index = VectorIndex(embedding_fn=embedding_fn or generate_embedding)
        self.bm25_index = BM25Index()

        # Initialize retriever
        reranker = create_reranker_fn() if use_reranking else None
        self.retriever = Retriever(self.bm25_index, self.vector_index, reranker_fn=reranker)

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the RAG system.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        all_chunks = []

        for i, doc in enumerate(documents):
            # Chunk the document
            chunks = self._chunk_document(doc)

            # Enrich with context if enabled
            if self.use_contextual_enrichment:
                chunks = enrich_chunks_with_context(chunks, full_document=doc)

            # Add metadata
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}

            for j, chunk in enumerate(chunks):
                chunk_doc = {
                    "content": chunk,
                    "doc_index": i,
                    "chunk_index": j,
                    **doc_metadata
                }
                all_chunks.append(chunk_doc)

        # Add to retriever (batch operation)
        self.retriever.add_documents(all_chunks)

    def _chunk_document(self, text: str) -> List[str]:
        """Chunk a document using the configured strategy."""
        if self.chunking_strategy == "characters":
            return chunk_by_characters(text)
        elif self.chunking_strategy == "sentences":
            return chunk_by_sentences(text)
        elif self.chunking_strategy == "sections":
            return chunk_by_section(text)
        elif self.chunking_strategy == "paragraphs":
            return chunk_by_paragraphs(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Search for relevant chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (chunk_document, relevance_score) tuples
        """
        return self.retriever.search(query, k=k)

    def query(self, question: str, k: int = 3) -> str:
        """Answer a question using RAG.

        Args:
            question: User's question
            k: Number of chunks to retrieve

        Returns:
            Claude's answer based on retrieved context
        """
        # Retrieve relevant chunks
        results = self.search(question, k=k)

        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Build context from retrieved chunks
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}]\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate answer with Claude
        prompt = f"""Answer the user's question based on the provided context. If the context doesn't contain enough information, say so.

<context>
{context}
</context>

<question>
{question}
</question>

Provide a clear, concise answer based on the context above."""

        messages = []
        add_user_message(messages, prompt)
        response = chat(messages)

        return text_from_message(response)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG System Examples")
    print("=" * 60)

    # Example documents
    sample_docs = [
        """
        Python is a high-level, interpreted programming language.
        It was created by Guido van Rossum and first released in 1991.
        Python emphasizes code readability with significant whitespace.
        It supports multiple programming paradigms including procedural,
        object-oriented, and functional programming.
        """,
        """
        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience. Common algorithms include
        linear regression, decision trees, random forests, and neural networks.
        Deep learning uses multi-layer neural networks for complex pattern recognition.
        """,
        """
        The Anthropic API provides access to Claude, a large language model.
        Claude can help with tasks like writing, analysis, coding, and math.
        The API supports features like streaming, tool use, and vision capabilities.
        Rate limits and pricing vary by model tier.
        """
    ]

    print("\n" + "=" * 60)
    print("Example 1: Basic RAG System")
    print("=" * 60)

    """
    # Uncomment to run
    rag = RAGSystem(
        chunking_strategy="sentences",
        use_contextual_enrichment=False,
        use_reranking=False
    )

    rag.add_documents(sample_docs)

    # Search
    results = rag.search("What is Python?", k=2)
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc['content'][:100]}...")
        print()

    # Query with answer generation
    answer = rag.query("What programming language did Guido van Rossum create?")
    print("Answer:", answer)
    """

    print("\n" + "=" * 60)
    print("Example 2: Advanced RAG with All Features")
    print("=" * 60)

    """
    # Uncomment to run
    advanced_rag = RAGSystem(
        chunking_strategy="sentences",
        use_contextual_enrichment=True,  # Add context to chunks
        use_reranking=True  # Use Claude for reranking
    )

    advanced_rag.add_documents(sample_docs)

    answer = advanced_rag.query("How does machine learning relate to AI?", k=3)
    print("Answer:", answer)
    """

    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    """
    # Uncomment to run
    # Create custom retriever
    vector_index = VectorIndex()
    bm25_index = BM25Index(k1=1.2, b=0.8)
    reranker = create_reranker_fn(top_k=5)

    retriever = Retriever(bm25_index, vector_index, reranker_fn=reranker)

    # Add documents with custom chunking
    chunks = chunk_by_paragraphs(sample_docs[0], paragraphs_per_chunk=1)
    enriched = enrich_chunks_with_context(chunks, full_document=sample_docs[0])

    docs = [{"content": chunk} for chunk in enriched]
    retriever.add_documents(docs)

    results = retriever.search("Python programming", k=2)
    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Content: {doc['content'][:150]}...")
        print()
    """

    print("\n" + "=" * 60)
    print("Pattern Summary:")
    print("=" * 60)
    print("""
Complete RAG System Architecture:

1. CHUNKING STRATEGIES
   - chunk_by_characters: Fixed-size chunks with overlap
   - chunk_by_sentences: Semantic boundaries (sentences)
   - chunk_by_section: Structure-based (headers, sections)
   - chunk_by_paragraphs: Natural paragraph boundaries

   Trade-offs:
   • Characters: Simple, but breaks semantic units
   • Sentences: Better boundaries, configurable size
   • Sections: Preserves structure, variable sizes
   • Paragraphs: Natural units, variable sizes

2. CONTEXTUAL ENRICHMENT
   - Prepend context to chunks before embedding
   - Context sources: document start + previous chunks
   - Uses Claude to generate succinct situating text
   - Improves embedding quality and retrieval accuracy

   Trade-offs:
   • Pro: Better retrieval, self-contained chunks
   • Con: Higher cost (Claude API calls), slower indexing

3. HYBRID SEARCH (VECTOR + BM25)
   - VectorIndex: Semantic similarity via embeddings
   - BM25Index: Lexical/keyword matching
   - Complementary strengths:
     • Vector: "car" matches "automobile"
     • BM25: Exact terms like "ERR_CODE_0x123"

4. RECIPROCAL RANK FUSION (RRF)
   - Combines rankings from multiple indexes
   - Score: sum(1 / (k + rank)) for each index
   - Rank-based (no score calibration needed)
   - Extensible to 3+ indexes

5. LLM RERANKING
   - Uses Claude's deep understanding
   - Applied to top candidates (cost optimization)
   - XML format for structured context
   - JSON response with document IDs

   Trade-offs:
   • Pro: Best accuracy, understands complex relevance
   • Con: Adds latency (~500ms-1s), higher cost

6. END-TO-END RAG
   - RAGSystem class orchestrates full pipeline
   - Configurable at every stage
   - query() method: retrieve + generate answer
   - search() method: retrieve only

CONFIGURATION GUIDELINES:

Basic RAG (Low cost, good performance):
- Sentence chunking
- No contextual enrichment
- Hybrid search only
- Use case: Large corpora, cost-sensitive

Standard RAG (Balanced):
- Sentence or paragraph chunking
- Contextual enrichment
- Hybrid search + RRF
- Use case: Most production applications

Premium RAG (Best accuracy):
- Semantic chunking (sentences/paragraphs)
- Contextual enrichment
- Hybrid search + RRF + Reranking
- Use case: High-stakes Q&A, accuracy critical

PERFORMANCE CONSIDERATIONS:

Indexing Time:
• Basic chunking: ~1ms per chunk
• Embeddings: ~50-200ms per batch (API latency)
• Contextual enrichment: +500-1000ms per chunk (Claude)
• Batch operations critical for performance

Query Time:
• Vector search: ~10-50ms (in-memory)
• BM25 search: ~5-20ms (in-memory)
• RRF fusion: ~1ms
• Reranking: +500-1000ms (Claude API)

Cost Optimization:
• Use Haiku for reranking/context (cheaper)
• Batch embedding calls (fewer API requests)
• Cache embeddings (avoid recomputation)
• Set reasonable k values (k=3-5 usually sufficient)

PRODUCTION TIPS:

1. Start simple, add complexity as needed
2. Measure retrieval quality before optimizing
3. Use batch operations for better throughput
4. Cache embeddings in persistent storage
5. Monitor API costs (embeddings + reranking)
6. Consider using production vector DBs (Pinecone, Chroma)
7. Implement retry logic for API failures
8. Add request/response logging for debugging

When to Use Each Component:
• Contextual enrichment: Small-medium corpora, accuracy matters
• BM25: Technical docs with specific terms/codes
• Vector search: General knowledge, semantic search
• Reranking: Top-k quality critical, budget allows
• RRF fusion: Always (minimal cost, significant benefit)
""")
