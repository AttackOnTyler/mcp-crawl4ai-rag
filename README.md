<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [ChromaDB](https://www.trychroma.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a local vector database (ChromaDB), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

## Tools

The server provides four essential web crawling and search tools:

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating contextual embeddings - optional)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -e .
   crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration (for contextual embeddings - optional)
# Get your Open AI API Key by following these instructions -
# https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key
OPENAI_API_KEY=your_openai_api_key

# The LLM you want to use for contextual embeddings (contextual retrieval)
# Leave this blank if you do not want to use contextual embeddings
# Generally this is a very cheap and fast LLM like gpt-4.1-nano
MODEL_CHOICE=

# ChromaDB Configuration
# Path where ChromaDB will store its data (defaults to ./chroma_db_mcp if not set)
# For Docker, this should be a path inside the container that is mounted as a volume (e.g., /app/chroma_data)
CHROMA_DB_PATH=./chroma_db_mcp

# Name of the ChromaDB collection to use (defaults to 'crawled_docs' if not set)
CHROMA_COLLECTION_NAME=crawled_docs
```

## Running the Server

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

## Running the Server

### Using Docker

To ensure your ChromaDB data persists across container restarts, you need to mount a volume.

```bash
# Create a local directory to store ChromaDB data (if it doesn't exist)
mkdir -p ./chroma_data_on_host

# Run the Docker container, mounting the local directory to the container's data path
docker run --env-file .env -p 8051:8051 -v ./chroma_data_on_host:/app/chroma_data mcp/crawl4ai-rag
```

> **Note:** The `-v ./chroma_data_on_host:/app/chroma_data` flag maps the `./chroma_data_on_host` directory on your host machine to the `/app/chroma_data` directory inside the container. Ensure the `CHROMA_DB_PATH` in your `.env` file is set to `/app/chroma_data` when running with this volume mount.

### Using Docker Compose

If you have Docker Compose installed, you can use the provided `docker-compose.yml` file for easier setup and management, including automatic volume creation for persistent data.

1.  Ensure you have a `.env` file in the project root based on the Configuration section.
2.  Run the following command in the project root:
    ```bash
    docker compose up --build -d
    ```

This command will build the Docker image (if necessary), create a named volume for ChromaDB data, and start the server in detached mode.

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "MODEL_CHOICE": "your_model_choice",
        "CHROMA_DB_PATH": "path/to/your/chroma_data",
        "CHROMA_COLLECTION_NAME": "crawled_docs"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "MODEL_CHOICE",
               "-e", "CHROMA_DB_PATH",
               "-e", "CHROMA_COLLECTION_NAME",
               "-v", "<host_path_to_chroma_data>:/app/chroma_data", # Mount volume for persistence
               "mcp/crawl4ai-rag"], # Corrected image name
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "MODEL_CHOICE": "your_model_choice",
        "CHROMA_DB_PATH": "/app/chroma_data", # Path inside the container
        "CHROMA_COLLECTION_NAME": "crawled_docs"
      }
    }
  }
}
```

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers
