# ContentForge AI

A Multi-Agent Content Intelligence Platform built using LangGraph, Pinecone, and LLMs to automate research, content generation, fact verification, SEO optimization, and quality evaluation.

## Overview

Traditional AI blog generators often produce content with hallucinations, weak research grounding, and limited quality control. ContentForge AI addresses these challenges through a multi-agent workflow that combines Retrieval-Augmented Generation (RAG), fact-checking, SEO analysis, and evaluation pipelines.

The system performs end-to-end content creation, from topic research to final quality assessment, while maintaining source-grounded generation using a vector database and semantic retrieval.

---

## Key Features

* Multi-Agent Workflow using LangGraph
* Retrieval-Augmented Generation (RAG)
* Pinecone Vector Database Integration
* Semantic Search using Sentence Transformers
* Automated Web Research using Tavily
* Parallel Section Generation with Writer Agents
* Hallucination Detection & Fact Verification
* SEO Optimization & Keyword Analysis
* Readability and Content Quality Assessment
* PDF Knowledge Base Ingestion
* Analytics Dashboard and Token Usage Tracking
* Source Citation Support

---

## Architecture

```text
User Topic
    ↓
Router Agent
    ↓
Research Agent
    ↓
RAG Layer (Pinecone)
    ↓
Orchestrator
    ↓
Parallel Writer Agents
    ↓
Reducer
    ↓
Fact Check Agent
    ↓
SEO Agent
    ↓
Reviewer Agent
    ↓
Evaluator Agent
    ↓
Final Blog
```

---

## Tech Stack

### AI & LLM

* OpenAI GPT-4.1 Mini
* LangGraph
* LangChain
* Sentence Transformers

### Retrieval & Search

* Pinecone Vector Database
* Tavily Search API

### Knowledge Processing

* PyMuPDF
* PDF Chunking
* Semantic Retrieval

### Frontend

* Streamlit

### Backend

* Python

---

## Workflow

### 1. Router Agent

Analyzes the topic and determines whether external research is required.

### 2. Research Agent

Collects relevant information from web sources using Tavily Search.

### 3. RAG Layer

Stores research evidence and uploaded documents as embeddings in Pinecone and retrieves relevant context during generation.

### 4. Orchestrator

Creates a structured content plan and generates section-level writing tasks.

### 5. Writer Agents

Generate blog sections in parallel using retrieved evidence and contextual information.

### 6. Reducer

Combines generated sections into a complete article.

### 7. Fact Check Agent

Extracts factual claims, verifies them against evidence, and generates factuality scores.

### 8. SEO Agent

Generates optimized titles, keywords, meta descriptions, and SEO scores.

### 9. Reviewer Agent

Evaluates readability, grammar, structure, and content quality.

### 10. Evaluator Agent

Measures faithfulness, relevance, and context utilization using an LLM-as-Judge approach.

---

## RAG Pipeline

```text
Research Sources / PDFs
          ↓
       Chunking
          ↓
      Embeddings
          ↓
       Pinecone
          ↓
   Similarity Search
          ↓
 Retrieved Context
          ↓
     Writer Agents
```

---

## PDF Knowledge Base

The platform supports ingestion of PDF documents and research papers.

Workflow:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Pinecone Storage
    ↓
Semantic Retrieval
```

This enables domain-specific content generation grounded in user-provided documents.

---

## Analytics

The dashboard tracks:

* SEO Score
* Readability Score
* Factuality Score
* Token Usage
* Generation Time
* Hallucination Risk
* Evaluation Metrics

---

## Example Use Cases

* AI Blog Generation
* Technical Content Creation
* Research-Based Article Writing
* Industry Reports
* Knowledge Base Driven Content Generation
* Research Paper Summarization
* SEO Content Production

---

## Future Enhancements

* Multi-modal RAG
* Human Feedback Loops
* Multi-Language Support
* Agent Memory
* Advanced Citation Tracking
* Self-Reflection Agents
* Knowledge Graph Integration

---

## Resume Highlights

* Built a multi-agent content intelligence platform using LangGraph for research, writing, fact verification, SEO optimization, and evaluation.
* Implemented Retrieval-Augmented Generation (RAG) using Pinecone Vector Database and SentenceTransformers embeddings.
* Developed hallucination detection and fact-checking pipelines with source-grounded verification.
* Integrated PDF knowledge-base ingestion, semantic retrieval, and analytics dashboards for content quality monitoring.
