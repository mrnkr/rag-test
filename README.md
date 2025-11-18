# RAG Test Project

A test project for exploring Retrieval-Augmented Generation (RAG) implementations.

## Overview

This project contains experiments and implementations related to RAG systems, which combine retrieval mechanisms with generative AI models to provide more accurate and contextual responses.

Built following [this tutorial](https://docs.langchain.com/oss/python/langchain/rag) and using only ollama models.

Tried multiple combinations for the chat agent and the embeddings and got the best results with the following combinations:

- Embeddings: `mxbai-embed-large` - Chat: `gpt-oss:20b` - Approach: RAG agents
- Embeddings: `mxbai-embed-large` - Chat: `llama3.1` - Approach: RAG chains

Since `llama3.1` is noticeably lighter than `gpt-oss:20b` and the number of queries sent to the model is reduced to one, the 'RAG chains' approach is more efficient and yields comparable results. Like the tutorial mentions, it's also less flexible.

## Getting Started

### Prerequisites

- Python 3.8+
- Required dependencies (see `requirements.txt`)

### Installation

```bash
git clone git@github.com:mrnkr/rag-test.git
cd rag-test
pip install -r requirements.txt
```

## Usage

`python src/main.py`

## Project Structure

```
rag-test/
├── README.md
├── requirements.txt
├── src/
|  ├── main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT
