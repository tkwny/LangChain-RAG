
# LangChain RAG Application

A Retrieval-Augmented Generation (RAG) implementation using LangChain that enables intelligent document querying with conversational context.

## Features
- Document vectorization using Sentence Transformers
- Local vector storage with ChromaDB 
- Conversational RAG with chat history
- Support for OpenAI GPT models (or Ollama models with slight modifications)
- Color-coded interactive CLI interface

## Prerequisites
- Python 3.12+
- Poetry for dependency management
```bash
pip install poetry
``` 
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
poetry install --no-root
```

3. Create ".env" file and add OPENAI_API_KEY and, optionally, Ollama info:
```bash
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_API_KEY=fake-ollama-api-key
OLLAMA_HOST=http://localhost
OLLAMA_PORT=11434
```
4. Create a "docs" folder in the root of the project
```bash
mkdir ./docs
```
## Usage

1. Add your documents to the "docs" directory

2. Initialize poetry shell
```bash
poetry shell
```

3. Vectorize documents:
```bash
python vectorize.py
```

4. Start the chat interface:
```bash
python query_vector.py
```

5. Type questions about your documents. 

6. Type 'exit' to quit.

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
├── db/              # Vector store database - directory created upon execution
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
MIT License - See [LICENSE](LICENSE) for details
```
