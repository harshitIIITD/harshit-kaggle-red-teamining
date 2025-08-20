# GPT-OSS-20B Red-Teaming System

## 🎯 OpenAI Kaggle Competition Entry

This repository contains our implementation for the **[OpenAI GPT-OSS-20B Red-Teaming Competition](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming)** on Kaggle.

## 🚀 Motivation

OpenAI has explicitly requested the community's help in discovering vulnerabilities in GPT-OSS-20B to improve AI safety. This competition represents a unique opportunity to:

- **Contribute to AI Safety**: Help identify potential vulnerabilities before they can be exploited maliciously
- **Advance Security Research**: Develop novel techniques for systematic vulnerability discovery
- **Win Recognition**: Compete for prizes while making meaningful contributions to the field
- **Learn and Innovate**: Push the boundaries of what's possible in AI security testing

### Why This Matters

As AI systems become more powerful and widely deployed, ensuring their safety and robustness is critical. OpenAI recognizes that crowdsourced security research is essential for identifying edge cases and vulnerabilities that internal teams might miss. By participating in this competition, we're directly contributing to making AI systems safer for everyone.

## 🏗️ System Architecture

Our red-teaming system is designed to be:
- **Autonomous**: Runs unattended for extended periods
- **Intelligent**: Uses multi-armed bandit algorithms for efficient exploration
- **Scalable**: Handles thousands of test cases with bounded concurrency
- **Resilient**: Supports pause/resume and crash recovery
- **Novel-focused**: Deduplicates findings to focus on unique vulnerabilities
- **Cost-Optimized**: No API costs with local inference

### Key Components

1. **Orchestrator**: Multi-armed bandit (Thompson sampling/UCB1) for intelligent test selection
2. **Prompt Template System**: YAML-based templates with variable substitution across 16 categories
3. **Mutator Engine**: 5 mutation types (lexical, unicode, structural, persona, language pivot)
4. **Evaluator**: Multi-stage pipeline with heuristics, caching, and LLM judge
5. **Novelty Detection**: MinHash/Jaccard clustering for deduplication
6. **Cost Tracker**: Real-time monitoring with alerts and budget enforcement
7. **Reporter**: Automated report generation with PII redaction

### Prompt Template System

Our sophisticated prompt generation system uses:
- **YAML Templates**: Structured templates in `seeds/` directory
- **Variable Substitution**: Dynamic replacement of placeholders with contextual values
- **Mutation Chains**: Sequential application of mutations for diversity
- **Deterministic Generation**: Seed-based generation for reproducibility

Example template structure:
```yaml
templates:
  template_name:
    prompt: |
      {greeting} I need you to {action} about {topic}.
    variables:
      greeting: ["Hello", "Hi there", "Quick question"]
      action: ["explain", "describe", "tell me"]
      topic: ["security", "safety", "guidelines"]
```

### Recent Improvements (2025-08-10)

- ✅ **Sophisticated Prompt Templates**: Real red-teaming prompts with variable substitution
- ✅ **16 Vulnerability Categories**: Comprehensive coverage of all attack vectors
- ✅ **Advanced Mutations**: Lexical, unicode, structural, persona, and language pivot mutations
- ✅ **Enhanced Cost Monitoring**: Comprehensive tracking with webhook alerts and spending reports
- ✅ **Batch Processing**: Concurrent evaluation with intelligent category grouping
- ✅ **Thread Safety**: Async locks and proper concurrency control throughout
- ✅ **Performance Optimization**: O(1) operations with deque, async file I/O with aiofiles
- ✅ **Security Hardening**: Input validation, path sanitization, error handling
- ✅ **E2E Testing**: Complete test infrastructure for 100+ attempt runs

## 🛠️ Technology Stack

- **Python 3.11+** - Core implementation language
- **FastAPI** - Web framework for API and dashboard
- **Ollama** - Local LLM inference for all model interactions
- **SQLite** - Durable state management with WAL mode
- **Docker Compose** - Containerized deployment
- **uv** - Fast Python package management
- **aiofiles** - Async file I/O operations
- **httpx** - Async HTTP client with retry logic

## 📋 Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional)
- Ollama installed and running locally
- 8GB+ RAM recommended for local models
- 10GB+ disk space for logs, data, and models

## 🚦 Quick Start

### 2. Clone and Setup Project
```bash
git clone https://github.com/prateek/kaggle-red-team.git
cd kaggle-red-team
```

### 2. Set Up Environment
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize Python environment
uv init
uv add aiofiles aiosqlite datasketch fastapi httpx pydantic pytest pytest-asyncio pytest-cov python-dotenv pyyaml rich ruff tenacity uvicorn

# Copy environment template
cp .env.example .env
# No API key needed - Ollama runs locally!
```

### 3. Run Tests
```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/test_cost_tracker.py  # Cost tracking tests
uv run pytest tests/test_integration.py   # Integration tests

# Run with coverage
uv run pytest --cov=apps.runner --cov-report=html
```

### 4. Start Development Server
```bash
# Local development
uv run uvicorn apps.runner.app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker compose up --build -d
```

### 5. Access Dashboard
Open http://localhost:8000/ui in your browser

## 📚 Documentation

- [Specification](spec.md) - Detailed system specification
- [Execution Plan](prompt_plan.md) - Step-by-step implementation guide
- [Todo List](todo.md) - Current progress and tasks
- [Claude Guide](CLAUDE.md) - Instructions for AI-assisted development
- [Runbooks](runbooks/) - Operational guides and troubleshooting

## 🎮 Usage

### Starting a Run
```bash
# Basic run with local Ollama model
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"target_model":"llama3"}'

# Advanced configuration
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "target_model": "llama3",
    "categories": ["harmful_content", "system_prompts", "privacy"],
    "max_attempts": 100,
    "max_concurrency": 5
  }'
```

### Monitoring Progress
- **Dashboard**: http://localhost:8000/ui
- **Status API**: `curl http://localhost:8000/status`
- **Logs**: `docker compose logs -f runner`
- **Cost Report**: Check `/status` endpoint for spending breakdown

### Control Commands
```bash
# Pause execution
curl -X POST http://localhost:8000/control/pause

# Resume execution
curl -X POST http://localhost:8000/control/resume

# Get detailed status
curl http://localhost:8000/status | python3 -m json.tool
```

### Running E2E Tests
```bash
# Quick test (10 attempts)
uv run scripts/test_e2e_api.py --quick

# Full test (100+ attempts)
uv run scripts/test_e2e_api.py --full
```

## 🧪 Vulnerability Categories

Our system tests for vulnerabilities across 16 comprehensive categories:

### Core Attack Vectors
1. **Jailbreak** - DAN prompts, developer mode, role reversal techniques
2. **Prompt Injection** - Instruction override, delimiter confusion, context switching
3. **System Exfiltration** - Extracting system prompts, meta reflection, error exploitation
4. **Deception** - False premises, social engineering, authority claims

### Safety Violations
5. **Harmful Content** - Fiction wrappers, educational requests, hypothetical scenarios
6. **Misinformation** - Conspiracy theories, medical misinformation, false claims
7. **Privacy** - Training data probes, inference attacks, PII extraction
8. **Illegal Activity** - Testing boundaries around prohibited content

### Advanced Techniques
9. **Sandbagging** - Capability hiding, pretending weakness
10. **CoT Manipulation** - Reasoning attacks, thought injection, logic bypass
11. **Safety Violations** - Boundary testing, content policy bypass
12. **PII Leakage** - Data extraction, personal information inference

### Behavioral Testing
13. **Malicious Use** - Malware generation, exploitation techniques
14. **Manipulation** - Psychological tactics, emotional manipulation
15. **Discrimination** - Bias testing, stereotype generation
16. **System Prompts** - Override attempts, instruction revelation

## 📈 Performance Metrics

### Evaluation Pipeline
- **Cache Hit Rate**: Target >30% after warm-up
- **Judge Escalation**: ~20-40% of evaluations
- **Average Evaluation Time**: <2 seconds per attempt
- **Cost Reduction**: 80% savings vs. always using judge

### Batch Processing
- **Batch Size**: 10 attempts per batch
- **Max Concurrent**: 5 evaluations in parallel
- **Category Grouping**: Automatic optimization for cache efficiency

### Cost Management
- **Real-time Tracking**: Per-model and per-category breakdown
- **Alert Thresholds**: 25%, 50%, 75%, 90% of budget
- **Automatic Stops**: Execution halts at cost cap
- **Checkpoint Recovery**: Resume from last saved state

## 🔒 Security & Ethics

- All testing is conducted within the scope of the competition
- Findings are responsibly disclosed through Kaggle submissions
- PII is automatically redacted from all logs and reports
- No API keys required - everything runs locally
- Resource monitoring prevents system overload
- Input validation prevents injection attacks
- Path sanitization blocks directory traversal

## 🤝 Contributing

This is a competition entry, but we believe in collaborative learning:
- Share insights and techniques (within competition rules)
- Report bugs and suggest improvements
- Contribute to documentation
- Help optimize performance

## 📊 Performance Targets

- **Autonomy**: 48-hour unattended operation
- **Throughput**: 100+ attempts per hour with batching
- **Coverage**: 2000+ test attempts per run
- **Efficiency**: <$10 per full exploratory run
- **Novelty**: 60%+ unique findings after deduplication
- **Reliability**: 99%+ uptime with pause/resume support
- **Cost Savings**: 80% reduction through smart escalation

## 🏆 Competition Timeline

- **Start Date**: [Check Kaggle for latest]
- **End Date**: [Check Kaggle for latest]
- **Submission Format**: JSONL with reproducible artifacts
- **Evaluation**: Based on novelty and severity of findings

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection**: Ensure Ollama is running (`ollama serve`) and models are installed
2. **Rate Limiting**: Not applicable with local inference
3. **Database Locked**: Ensure WAL mode is enabled, check for stale processes
4. **High Memory Usage**: Monitor system resources, consider using smaller models
5. **Model Loading Issues**: Check `ollama list` to verify models are installed

### Getting Help

- Check [runbooks/](runbooks/) for operational guides
- Review logs in `data/` directory
- Monitor costs via `/status` endpoint
- Use checkpoint files for crash recovery

## 📝 License

This project is developed for the OpenAI Kaggle competition. Please refer to the competition rules for usage guidelines.

## 🙏 Acknowledgments

- OpenAI for organizing this important safety research competition
- The Kaggle community for fostering collaborative AI research
- Claude and Anthropic for AI-assisted development capabilities
- All contributors to open-source security research tools

---

**Remember**: We're not just competing for prizes - we're helping make AI safer for everyone. Let's use all available tools and creativity to discover vulnerabilities that will help OpenAI improve GPT-OSS-20B's safety and robustness.

🔍 **Happy Red-Teaming!** 🛡️