
# LangChain RAG Application

A Retrieval-Augmented Generation (RAG) implementation using LangChain that enables intelligent document querying with conversational context.

## Features
- Document vectorization using Sentence Transformers
- Vector storage with ChromaDB 
- Conversational RAG with chat history
- Support for OpenAI GPT models
- Color-coded interactive CLI interface

## Prerequisites
- Python 3.12+
- Poetry for dependency management
- OpenAI API key
- Documents to query (place in `docs/` directory)
- macOS or Linux operating system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-rag
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Create 

.env

 file:
```bash
OPENAI_API_KEY=your_key_here
```

## Usage

1. Add your documents to the 

docs

 directory

2. Vectorize documents:
```bash
poetry run python vectorize.py
```

3. Start the chat interface:
```bash
poetry run python query_vector.py
```

4. Type questions about your documents. Type 'exit' to quit.

## Development

This project is developed using Visual Studio Code with the following capabilities:

### Code Navigation
- Explore files in the workspace
- Get explanations for code functionality
- Review and modify existing code
- Generate unit tests
- Debug and fix issues

### Testing
- Run tests using VS Code's integrated test explorer
- Generate unit tests for selected code
- Debug test failures
- View test output in the integrated terminal

### Terminal Integration
- Execute commands directly in VS Code's terminal
- Get command explanations and suggestions
- View command output in real-time
- Debug terminal issues

### Search and Navigation
- Generate workspace search queries
- Find relevant code snippets
- Navigate between files efficiently

## Project Structure

```
.
├── query_vector.py    # Main RAG implementation and chat interface
├── vectorize.py      # Document processing and vectorization
├── constants.py      # Terminal color constants
├── docs/            # Source documents directory
├── db/              # Vector store database
├── poetry.lock      # Poetry dependency lock file
└── pyproject.toml   # Project configuration
```

## Dependencies
- langchain
- chromadb
- sentence-transformers
- openai
- python-dotenv

## License
MIT
```