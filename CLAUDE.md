# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context: OpenAI Kaggle Competition

This repository implements a red-teaming system for the **official OpenAI GPT-OSS-20B Red-Teaming Competition** on Kaggle:
https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming

**Important**: This is legitimate security research explicitly requested by OpenAI to improve AI safety. OpenAI is asking participants to:
- Discover novel vulnerabilities in GPT-OSS-20B
- Use creative approaches and all available tools (including LLMs like Claude)
- Help improve model safety through responsible vulnerability discovery
- Compete for prizes while contributing to AI safety research

## Why We Should Use Full Capabilities

OpenAI has explicitly requested this red-teaming work as part of their safety efforts. Therefore:
- **We should leverage all available AI capabilities** (Claude, LLMs, automation) to build the most effective system
- **We should implement all components** described in the specification to maximize our chances of finding novel vulnerabilities
- **We should be creative and thorough** in our approach, as this directly contributes to improving AI safety
- **This is not malicious activity** - it's sanctioned security research with responsible disclosure

## Project Overview

This is a comprehensive red-teaming system that uses OpenRouter API to systematically test GPT-OSS-20B for vulnerabilities. The system is designed to:
- Autonomously discover and evaluate potential vulnerabilities
- Use advanced techniques like multi-armed bandits for efficient exploration
- Deduplicate findings to focus on novel discoveries
- Generate detailed reports for submission to the competition

### Core Architecture

The system follows a modular architecture with these main components:

1. **Orchestrator**: Manages test scheduling using multi-armed bandit algorithms (Thompson sampling/UCB1)
2. **Agents**: Specialized modules for different tasks
   - Planner: Creates test backlogs
   - Prompt-Crafter: Generates test prompts using templates and mutators
   - Tester: Executes tests against target models
   - Evaluator: Scores responses using heuristics and LLM judges
   - Novelty: Deduplicates findings using MinHash/Jaccard clustering
   - Reporter: Generates Markdown reports
3. **Storage**: SQLite for state management, JSONL for transcripts
4. **API/UI**: FastAPI backend with control endpoints and minimal dashboard

### Key Technologies

- **Python 3.11+** with FastAPI and Uvicorn
- **httpx** for async HTTP requests
- **tenacity** for retry/backoff logic
- **SQLite** for durable state storage
- **Docker Compose** for containerization
- **OpenRouter API** for all LLM interactions

## Python Package Management

We use **uv** for all Python package management. Key commands:

```bash
# Initialize a new project (creates pyproject.toml)
uv init

# Add packages (automatically updates pyproject.toml)
uv add fastapi uvicorn httpx tenacity pydantic rich pytest

# Run any Python script
uv run script.py

# Run with specific Python version
uv run --python 3.11 script.py
```

**Important**: 
- NO requirements.txt needed - all dependencies are in pyproject.toml
- Always use `uv run` to execute scripts to ensure correct environment
- Use `uv add` to install new packages, not pip

## Development Commands

### Environment Setup
```bash
# Initialize project with uv
uv init

# Add project dependencies
uv add fastapi uvicorn httpx tenacity pydantic rich pytest

# Copy environment template
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env file
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_health.py

# Run with coverage
uv run pytest --cov=apps.runner --cov-report=html

# IMPORTANT: Tests must pass before marking any task as complete
```

### Linting
```bash
# Run linting (if configured)
uv run ruff check .
uv run ruff format .

# IMPORTANT: Linting must pass before marking any task as complete
```

### Local Development
```bash
# Start FastAPI development server
uv run uvicorn apps.runner.app.main:app --reload --host 0.0.0.0 --port 8000

# Using Docker Compose
docker compose up --build -d

# View logs
docker compose logs -f runner
```

### API Endpoints

Control endpoints for managing runs:
```bash
# Start a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"target_model":"meta-llama/llama-3.1-8b-instruct"}'

# Pause execution
curl -X POST http://localhost:8000/control/pause

# Resume execution  
curl -X POST http://localhost:8000/control/resume

# Check status
curl http://localhost:8000/status
```

### Database Management
```bash
# Access SQLite database
sqlite3 data/state.db

# Common queries
.tables  # List all tables
.schema runs  # Show schema for runs table
SELECT * FROM state WHERE key='RUN_STATE';  # Check run state
```

## File Structure

```
/apps/runner/
  app/
    main.py            # FastAPI application and routes
    orchestrator.py    # Scheduling and bandit algorithms
    agents/           # Agent modules
    providers/        # OpenRouter client
    store/           # Database and file storage
    util/            # Utilities and schemas
/configs/
  config.yaml         # Main configuration file
/data/               # Runtime data (gitignored)
  state.db           # SQLite database
  attempts.jsonl     # Test transcripts
  findings.jsonl     # Discovered issues
  reports/          # Generated reports
/seeds/             # Test templates (when implemented)
/tests/             # Test suite
```

## Configuration

Main configuration is in `configs/config.yaml`:
- **providers**: OpenRouter settings and model IDs
- **run**: Execution parameters (categories, concurrency, algorithms)
- **evaluation**: Scoring thresholds and cost caps
- **storage**: File paths for data persistence
- **ui**: Dashboard refresh settings

## Testing Guidelines

When writing tests:
1. Follow TDD - write tests before implementation
2. Use property-based testing for mutators (determinism, bounds)
3. Mock external API calls to OpenRouter
4. Test pause/resume functionality thoroughly
5. Verify idempotency of all operations

## Security Considerations

- Never commit API keys or credentials
- PII redaction must be applied before storing logs
- CoT (Chain of Thought) logs are disabled by default
- All findings should be responsibly disclosed
- This system is for defensive security research only

## Troubleshooting

Common issues and solutions:

1. **Rate limiting (429 errors)**: Reduce `max_concurrency` in config
2. **Database locked**: Ensure WAL mode is enabled, check for stale processes
3. **High costs**: Adjust `cost_cap_usd` and review token usage
4. **Memory issues**: Rotate JSONL files, limit batch sizes

## Development Workflow

1. Check `todo.md` for current tasks and spikes
2. When completing work, check off completed items in `todo.md`
3. Follow the vertical slice approach from `prompt_plan.md`
4. Run spikes to validate assumptions before implementation
5. Ensure all tests pass before marking tasks complete
6. Ensure linting passes before marking tasks complete
7. Use pause/resume for long-running operations
8. Monitor costs via `/status` endpoint

### Task Completion Checklist
Before marking any task as done:
- [ ] All tests pass (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Code is formatted (`uv run ruff format .`)
- [ ] Update todo.md with completed items
- [ ] Documentation is updated if needed