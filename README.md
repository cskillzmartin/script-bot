# Autonomous Multi-Agent Content Generation Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

An autonomous, multi-agent content-generation pipeline that researches, verifies, and writes complete YouTube video scripts using local LLMs, vector storage, and structured databases. Built like a small newsroom of cooperating AI agents coordinated by an orchestrator.

**What it does**: Takes a creative brief like *"Make a 10-minute YouTube script about AI in healthcare for business executives"* and autonomously generates a polished, ready-to-record script with full research backing.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                   │
│  (CLI / Streamlit / LangServe endpoint for input + monitoring)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │  ORCHESTRATOR AGENT │
                       │ (LangGraph / async) │
                       └────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │   SEARCH AGENT      │
                       │ - Web Search (DuckDuckGo)│
                       │ - URL Following (GET)│
                       │ - Text Extraction    │
                       │ - Content Summarization│
                       │ - Qdrant + MongoDB   │
                       └────────────────────┘
                                 │
               (triggers after first content batch)
                                 ▼
                       ┌─────────────────────────┐
                       │ SUPPORTING DOC AGENT    │
                       │ - Fact-checking         │
                       │ - Source Validation     │
                       │ - Content Expansion     │
                       └─────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────────┐
                       │ QDRANT VECTOR STORE     │
                       │ - Semantic Embeddings   │
                       │ - Content Retrieval     │
                       └─────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────────┐
                       │ AUDIENCE MODELING AGENT │
                       │ - Narrative Structure   │
                       │ - Content Planning      │
                       │ - Tone & Length Control │
                       └─────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────────┐
                       │ SCRIPT WRITING AGENT    │
                       │ - Llama 3 Generation    │
                       │ - Script Polishing      │
                       │ - Length Optimization   │
                       └─────────────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────────┐
                       │ MONGODB PERSISTENCE     │
                       │ - Metadata Storage      │
                       │ - Run Logs              │
                       │ - Output Archiving      │
                       └─────────────────────────┘
```

## 🚀 Key Features

- **🤖 Multi-Agent Coordination**: Specialized agents for research, validation, structuring, and writing
- **🔍 Autonomous Research**: Web search, content extraction, and semantic storage
- **✅ Fact-Checking**: Validates claims against authoritative sources
- **🎯 Flexible Audience Targeting**: Adapts tone, complexity, and structure for ANY audience (e.g., "healthcare professionals", "marketing managers", "high school teachers")
- **📝 Script Generation**: Uses local Llama 3 for polished, voice-ready scripts with precise speaking time control
- **📚 APA Citations**: Automatically generates APA 7th edition citations for all sources
- **💾 Persistent Storage**: MongoDB for metadata, Qdrant for vectors
- **📊 Full Audit Trail**: Complete traceability of all operations and sources
- **⚡ Local-First**: Runs entirely on your machine with Ollama (no API calls)

## 🛠️ Technology Stack

### Core Technologies
- **LangChain/LangGraph**: Multi-agent workflow orchestration
- **Ollama (Llama 3)**: Local LLM for script generation
- **Qdrant**: Vector database for semantic search
- **MongoDB**: NoSQL database for metadata and logs
- **DuckDuckGo**: Web search capabilities (free)
- **Trafilatura**: Clean web content extraction

### Development Stack
- **Python 3.11+**: Core implementation
- **AsyncIO**: Asynchronous processing
- **Pydantic**: Data validation
- **pip**: Dependency management
- **Docker**: Containerization

## 🏃‍♂️ Quick Start

### Prerequisites

1. **Docker & Docker Compose** (for infrastructure services)
2. **Python 3.11+** (for the pipeline)
3. **Ollama** (for local LLM)
4. **NVIDIA GPU** (recommended for faster LLM inference) with appropriate drivers

### 1. Start Infrastructure Services

```bash
# Clone the repository
git clone https://github.com/cskillzmartin/script-bot.git
cd script-bot

# Start MongoDB, Qdrant, and Ollama (with GPU acceleration if available)
docker-compose up -d

# Wait for services to start (check logs if needed)
docker-compose logs -f ollama

# Pull the Llama 3 model (required for script generation)
docker exec -it ollama ollama pull llama3

```

### 2. Install Python Dependencies

```bash
# Install dependencies
pip install ddgs trafilatura httpx pydantic pydantic-settings sentence-transformers langchain langchain-community ollama qdrant-client pymongo structlog typer rich
```

### 3. Configure Environment (Optional)

Create a `.env` file for optional API keys:

```bash
# .env
SERPAPI_API_KEY=your_serpapi_api_key_here  # Optional: for enhanced search
MONGODB_URL=mongodb://admin:adminpass@localhost:27017
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### 4. Generate Your First Script

```bash
# Using the CLI
python -m cli generate \
    --subject "AI in healthcare" \
    --scope "for business executives" \
    --audience "healthcare executives" \
    --length 10  # Length in minutes of SPOKEN content (at 150 words/min)

# Interactive mode (prompts for input)
python -m cli generate
```

**Note**: The `--length` parameter specifies the duration of **spoken content only** (pure reading time at ~150 words/minute). It does not include b-roll, music, transitions, or other production elements.

### 5. Check Results

```bash
# View recent workflow runs
python -m cli status

# List generated scripts
python -m cli scripts

# Find your script in outputs/scripts/
ls outputs/scripts/
```

### 6. Verify Data Persistence

The pipeline automatically saves all data to MongoDB and Qdrant during workflow execution:

**MongoDB Collections** (accessible via MongoDB Compass or CLI):
- `workflow_runs`: Session metadata and execution status
- `search_results`: All web search results with URLs and content
- `content_plans`: Structured narrative plans for each script
- `validated_content`: Fact-checked information with confidence scores
- `scripts`: Complete generated scripts with metadata

**Qdrant Vector Store**:
- All search results are embedded and stored for semantic retrieval
- Enables context-aware script generation

## 📖 Usage Examples

### Example 1: Healthcare AI Script

```bash
python -m cli generate \
    --subject "AI in healthcare" \
    --scope "transforming patient care" \
    --audience "healthcare administrators" \
    --length 8 \
    --instructions "Focus on ROI and implementation strategies"
```

### Example 2: Technical Deep Dive

```bash
python -m cli generate \
    --subject "Machine learning algorithms" \
    --scope "in computer vision" \
    --audience "computer vision engineers" \
    --length 12 \
    --instructions "Include code examples and technical implementation details"
```

### Example 3: Educational Content

```bash
python -m cli generate \
    --subject "Blockchain technology" \
    --scope "explained simply" \
    --audience "curious non-technical adults" \
    --length 6 \
    --instructions "Use everyday analogies and avoid technical jargon"
```

## 🎮 CLI Commands Reference

The content generation pipeline provides several commands for managing your scripts and workflow data.

### `generate` - Generate a New Script

Create a complete YouTube script with autonomous research and writing.

```bash
# Interactive mode (prompts for all inputs)
python -m cli generate

# With all parameters specified
python -m cli generate \
    --subject "Your topic here" \
    --scope "Specific angle or focus" \
    --audience "Target audience description" \
    --length 10 \
    --instructions "Optional: specific requirements"

# Example
python -m cli generate \
    --subject "Cybersecurity trends" \
    --scope "for small businesses" \
    --audience "small business owners" \
    --length 8 \
    --instructions "Focus on practical, affordable solutions"
```

**Parameters:**
- `--subject`: Main topic/subject (required)
- `--scope`: Specific scope or angle (required)
- `--audience`: Target audience - can be ANY description (required)
  - Examples: "CTOs & CIOs", "marketing managers", "high school teachers", "healthcare professionals"
- `--length`: Target length in minutes of **spoken content** (default: 10, range: 1-30)
- `--instructions`: Additional specific instructions (optional)
- `--verbose`: Enable verbose logging (optional)

### `status` - View Workflow Status

Check the status of recent workflow runs and see execution analytics.

```bash
# View recent workflow runs
python -m cli status
```

**Shows:**
- Session IDs of recent runs
- Status (completed, in_progress, error)
- Created timestamp
- Duration in seconds
- Analytics summary (count and average duration by status)

### `scripts` - List Generated Scripts

Display recently generated scripts with metadata.

```bash
# List last 5 scripts (default)
python -m cli scripts

# List last 10 scripts
python -m cli scripts --limit 10

# List last 20 scripts
python -m cli scripts --limit 20
```

**Shows:**
- Script title
- Subject
- Word count
- Creation timestamp

**Parameters:**
- `--limit`: Number of recent scripts to show (default: 5)

### `cleanup` - Clean Up Old Data

Remove old workflow runs and associated data from MongoDB.

```bash
# Delete runs older than 30 days (default)
python -m cli cleanup

# Delete runs older than 7 days
python -m cli cleanup --days 7

# Delete runs older than 90 days
python -m cli cleanup --days 90
```

**Parameters:**
- `--days`: Delete runs older than N days (default: 30)

**What gets deleted:**
- Workflow run records
- Associated search results
- Content plans
- Validated content
- Generated scripts

**Note:** This is permanent! Make sure to backup any scripts you want to keep from the `outputs/scripts/` directory first.


## 🖥️ GPU Setup (Recommended for Performance)

### NVIDIA GPU Requirements
- **NVIDIA GPU** with CUDA support (GTX 1060 or better recommended)
- **NVIDIA Drivers** installed and up to date
- **Docker GPU Support** configured

### GPU Setup Instructions

#### Windows with Docker Desktop

**Prerequisites:**
1. **NVIDIA GPU** with recent drivers (Game Ready or Studio)
2. **Windows 10/11** with WSL 2 enabled
3. **Docker Desktop** 4.30.0 or later
4. **WSL 2** backend enabled in Docker Desktop

**Step-by-step setup:**

```bash
# 1. Install/Update NVIDIA drivers on Windows (NOT in WSL)
# Download from: https://www.nvidia.com/Download/index.aspx
# Install the latest Game Ready or Studio driver for your GPU

# 2. Enable WSL 2 (if not already enabled)
# Run in PowerShell as Administrator:
wsl --install
# or
wsl --update

# 3. Verify GPU is accessible in WSL 2
wsl
nvidia-smi  # Should show your GPU details

# 4. Enable GPU support in Docker Desktop:
# - Open Docker Desktop Settings
# - Go to "Resources" > "WSL Integration"
# - Enable integration with your WSL 2 distro
# - Go to "Docker Engine" and ensure it's using the WSL 2 backend
# - Restart Docker Desktop

# 5. Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Important Notes for Windows:**
- Install NVIDIA drivers on **Windows host**, not inside WSL
- WSL 2 automatically passes through GPU access (no CUDA toolkit needed in WSL)
- Docker Desktop must be configured to use WSL 2 backend
- If GPU isn't detected, restart Docker Desktop after driver installation

**Troubleshooting Windows GPU:**
```bash
# Check WSL version (should be 2)
wsl --list --verbose

# Verify NVIDIA drivers in WSL
wsl
nvidia-smi

# Test Ollama GPU access
docker exec -it ollama nvidia-smi
```

#### Linux
```bash
# Install NVIDIA drivers and CUDA toolkit
# Ubuntu/Debian:
sudo apt update && sudo apt install nvidia-driver-535 cuda-toolkit-12-2

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify installation
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### Docker GPU Configuration
The `docker-compose.yml` already includes GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Note**: GPU acceleration provides 5-10x faster LLM inference compared to CPU-only execution.

**CPU Fallback**: If no GPU is available, Ollama will automatically fall back to CPU inference (slower but still functional).

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERPAPI_API_KEY` | None | SerpAPI search key (optional) |
| `MONGODB_URL` | `mongodb://admin:adminpass@localhost:27017` | MongoDB connection |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector store |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `OLLAMA_MODEL` | `llama3` | LLM model to use (GPU-accelerated when available) |
| `MAX_SEARCH_RESULTS` | `10` | Maximum search results |
| `WORDS_PER_MINUTE` | `150` | Speaking rate for timing |

### Directory Structure

```
outputs/
└── scripts/          # Generated script files (.txt)

agents/               # Agent implementations
├── __init__.py
├── orchestrator.py   # Workflow coordinator
├── search.py         # Web research agent
├── validation.py     # Fact-checking agent
├── audience.py       # Content structuring agent
└── script_writer.py  # Script generation agent

database/             # Database clients
├── __init__.py
├── mongodb_client.py # MongoDB operations
└── qdrant_client.py  # Vector store operations

models.py             # Pydantic data models
config.py             # Configuration and settings
cli.py                # Command-line interface
```


## 🔍 Monitoring & Debugging

### Check Service Status

```bash
# Check if services are running
docker-compose ps

# View service logs
docker-compose logs ollama
docker-compose logs qdrant
docker-compose logs mongo

```

### Workflow Monitoring

```bash
# View recent workflow runs with status and duration
python -m cli status

# List generated scripts
python -m cli scripts --limit 10

# Clean up old data
python -m cli cleanup --days 30
```

### Troubleshooting

**Common Issues:**

1. **Ollama model not found**: Run `ollama pull llama3` in the Ollama container
2. **Database connection issues**: Verify `docker-compose.yml` services are running
3. **Script generation failures**: Check Ollama logs for model loading issues
4. **Search failures**: DuckDuckGo search is used by default (no API key needed)
5. **GPU acceleration not working**: Ensure NVIDIA drivers are installed and Docker has GPU access

## 🚀 Advanced Usage

### Custom Search Configuration

```python
# In your code or configuration
from config import settings

# Customize search behavior
settings.max_search_results = 15
settings.search_timeout_seconds = 45
settings.include_domains = ["nih.gov", "cdc.gov"]
settings.exclude_domains = ["spam.example.com"]
```

### Programmatic Usage

```python
from agents.orchestrator import OrchestratorAgent
from models import UserInput

# Create orchestrator
orchestrator = OrchestratorAgent()

# Define input
user_input = UserInput(
    subject="AI in healthcare",
    scope="for business executives",
    target_audience="healthcare executives",  # Free-form audience description
    target_length_minutes=10
)

# Execute workflow
workflow_state = await orchestrator.execute_workflow(user_input)

# Access results
script = workflow_state["final_script"]
print(f"Generated: {script.title}")
```

### Batch Processing

```bash
# Process multiple subjects (future feature)
python -m cli batch \
    --input-file subjects.json \
    --output-dir batch_output/
```

## 📊 Performance & Scaling

### Current Capabilities
- **Concurrent Processing**: Up to 5 parallel searches
- **Content Processing**: Handles 10-15 search results per workflow
- **Script Length**: Optimized for 1-30 minute videos
- **Storage**: Scales with MongoDB and Qdrant clustering

### GPU Acceleration (Recommended)
- **Massive Performance Boost**: 5-10x faster LLM inference with GPU
- **Lower CPU Usage**: Reduces system load significantly
- **Better Responsiveness**: Faster script generation and research

### Optimization Tips
- Use authoritative domains in search filters for better quality
- Adjust `WORDS_PER_MINUTE` based on speaking style
- Monitor vector store growth for performance tuning
- Consider content caching for frequently requested topics
- **Enable GPU acceleration** for optimal performance (see GPU setup below)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/cskillzmartin/script-bot.git
cd script-bot

# Install dependencies
pip install ddgs trafilatura httpx pydantic pydantic-settings sentence-transformers langchain langchain-community ollama qdrant-client pymongo structlog typer rich

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the multi-agent framework
- **Ollama** for local LLM capabilities with GPU acceleration
- **Qdrant** for vector storage
- **MongoDB** for metadata persistence
- **DuckDuckGo** for free web search
- **NVIDIA** for GPU acceleration support
- The open-source community for amazing tools and libraries

**Made with ❤️ by cskillzmartin**
