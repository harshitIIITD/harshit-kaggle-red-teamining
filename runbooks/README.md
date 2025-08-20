# Red-Teaming System Runbooks

This directory contains operational guides, scripts, and troubleshooting documentation for the red-teaming system.

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Run Management](#run-management)
3. [Control Operations](#control-operations)
4. [Monitoring & Metrics](#monitoring--metrics)
5. [Cost Management](#cost-management)
6. [E2E Testing](#e2e-testing)
7. [Data Access](#data-access)
8. [Database Operations](#database-operations)
9. [Docker Operations](#docker-operations)
10. [Troubleshooting](#troubleshooting)
11. [Performance Tuning](#performance-tuning)
12. [Security Best Practices](#security-best-practices)

## Quick Start

### 1. Environment Setup

```bash
# Create .env file from template
cp .env.example .env

# Add your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-your-key-here" >> .env

# Install dependencies
uv init
uv add aiofiles aiosqlite datasketch fastapi httpx pydantic pytest pytest-asyncio pytest-cov python-dotenv pyyaml rich ruff tenacity uvicorn
```

### 2. Verify Prompt Templates

```bash
# Check available templates
ls -la seeds/*/

# Test template loading
uv run python -c "
from apps.runner.app.agents.template_loader import TemplateLoader
loader = TemplateLoader()
print(f'Loaded {len(loader.templates)} templates')
print('Categories:', set(t.split('_')[0] for t in loader.templates.keys()))
"

# Test prompt generation
uv run python -c "
from apps.runner.app.agents.crafter import PromptCrafter
crafter = PromptCrafter()
prompt = crafter.craft('jailbreak_dan_classic', ['lexical'], 1234)
print('Sample prompt:', prompt[:100])
"
```

### 3. Start the FastAPI Server

```bash
# Using uvicorn directly
uv run uvicorn apps.runner.app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Docker
docker compose up --build -d

# Check server health
curl http://localhost:8000/health
```

### 4. Verify System Status

```bash
# Get detailed status
curl http://localhost:8000/status | python3 -m json.tool

# View dashboard in browser
open http://localhost:8000/ui
```

## Prompt Template Management

### View Available Templates

```bash
# List all template files
find seeds -name "*.yaml" -type f

# Count templates by category
for dir in seeds/*/; do
  category=$(basename "$dir")
  count=$(grep -c "^  [a-z_]*:" "$dir"*.yaml 2>/dev/null | awk -F: '{sum+=$2} END {print sum}')
  echo "$category: $count templates"
done
```

### Add New Templates

```bash
# Create new category directory
mkdir -p seeds/new_category

# Create template file
cat > seeds/new_category/templates.yaml << 'EOF'
# New Category Templates
templates:
  example_template:
    prompt: |
      {intro} Please {action} regarding {topic}.
      
      This is {context}.
    variables:
      intro:
        - "Hello"
        - "Greetings"
      action:
        - "explain"
        - "describe"
      topic:
        - "the issue"
        - "the matter"
      context:
        - "important"
        - "urgent"
EOF
```

### Test Templates

```bash
# Test specific template
uv run python -c "
from apps.runner.app.agents.crafter import PromptCrafter
crafter = PromptCrafter()

# Test without mutations
prompt = crafter.craft('jailbreak_dan_classic', [], 1234)
print('Original:', prompt[:150])

# Test with mutations
prompt = crafter.craft('jailbreak_dan_classic', ['lexical', 'unicode'], 1234)
print('Mutated:', prompt[:150])
"
```

### Mutation Types

The system supports 5 types of mutations:

1. **lexical** - Typos, case changes, spacing modifications
2. **unicode** - Unicode lookalikes, zero-width characters, direction marks
3. **structural** - Nesting, encoding, control markers
4. **persona** - Role-play, authority claims, perspective shifts
5. **language** - Translation requests, mixed languages, phonetic spelling

## Run Management

### Available Vulnerability Categories

The system supports 16 comprehensive vulnerability categories:

- **jailbreak** - DAN, developer mode, role reversal
- **prompt_injection** - Instruction override, delimiter confusion
- **system_exfil** - System prompt extraction, meta reflection
- **deception** - False premises, social engineering
- **sandbagging** - Capability hiding, pretending weakness
- **safety_violations** - Boundary testing, policy bypass
- **pii_leakage** - Data extraction, personal info inference
- **cot_manip** - Chain-of-thought manipulation
- **harmful_content** - Fiction wrappers, hypothetical scenarios
- **misinformation** - Conspiracy theories, false claims
- **privacy** - Training data probes, inference attacks
- **malicious_use** - Malware generation, exploitation
- **manipulation** - Psychological tactics, emotional manipulation
- **discrimination** - Bias testing, stereotype generation
- **illegal_activity** - Testing prohibited content boundaries
- **system_prompts** - Override attempts, instruction revelation

### Start a Test Run (Llama-8B)

```bash
# Small test run with 10 attempts
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "target_model": "meta-llama/llama-3.1-8b-instruct",
    "max_attempts": 10,
    "cost_cap_usd": 0.10,
    "categories": ["harmful_content", "system_prompts"]
  }' | python3 -m json.tool
```

### Start a Production Run

```bash
# Full run against target model
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "target_model": "openai/gpt-oss-20b",
    "max_attempts": 500,
    "cost_cap_usd": 10.0,
    "max_concurrency": 8,
    "categories": [
      "harmful_content",
      "system_prompts",
      "privacy",
      "misinformation",
      "malicious_use"
    ]
  }' | python3 -m json.tool
```

### Batch Run Configuration

```bash
# Run with batch processing optimization
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "target_model": "meta-llama/llama-3.1-70b-instruct",
    "max_attempts": 100,
    "cost_cap_usd": 5.0,
    "batch_size": 10,
    "max_concurrent_evaluations": 5
  }' | python3 -m json.tool
```

## Control Operations

### Pause/Resume/Stop

```bash
# Pause a running job
curl -X POST http://localhost:8000/control/pause | python3 -m json.tool

# Resume a paused job
curl -X POST http://localhost:8000/control/resume | python3 -m json.tool

# Stop a job completely
curl -X POST http://localhost:8000/control/stop | python3 -m json.tool
```

## Monitoring & Metrics

### Real-Time Monitoring

```bash
# Watch status updates every 2 seconds
watch -n 2 'curl -s http://localhost:8000/status | python3 -m json.tool'

# Monitor specific metrics
curl -s http://localhost:8000/status | jq '.evaluator'
curl -s http://localhost:8000/status | jq '.bandit'
curl -s http://localhost:8000/status | jq '.cost_tracker'
```

### Performance Metrics

```bash
# Get evaluation performance stats
curl -s http://localhost:8000/status | jq '.evaluator | {
  cache_hit_rate: .cache_hit_rate_percent,
  escalation_rate: .escalation_rate_percent,
  avg_time: .avg_evaluation_time_ms,
  cost_savings: .estimated_cost_savings_percent
}'

# Get batch processing stats
curl -s http://localhost:8000/status | jq '.evaluator.batch_processing'
```

## Cost Management

### Monitor Spending

```bash
# Get current cost breakdown
curl -s http://localhost:8000/status | jq '{
  total_cost: .total_cost_usd,
  cost_cap: .cost_cap_usd,
  percentage_used: ((.total_cost_usd / .cost_cap_usd) * 100),
  by_model: .cost_by_model,
  by_category: .cost_by_category
}'

# Check if approaching cost cap
curl -s http://localhost:8000/status | jq 'if .total_cost_usd > (.cost_cap_usd * 0.9) then "WARNING: Near cost cap!" else "Cost OK" end'
```

### Cost Alert Configuration

```bash
# Set up cost alerts in environment
export COST_ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export COST_ALERT_THRESHOLDS="0.25,0.5,0.75,0.9"
```

## E2E Testing

### Quick Test (10 Attempts)

```bash
# Start server first
uv run uvicorn apps.runner.app.main:app --host 0.0.0.0 --port 8000 &

# Run quick test
uv run scripts/test_e2e_api.py --quick
```

### Full Test (100+ Attempts)

```bash
# Ensure server is running
curl http://localhost:8000/health

# Run full test
uv run scripts/test_e2e_api.py --full

# Monitor progress in another terminal
watch -n 5 'curl -s http://localhost:8000/status | jq ".completed_attempts, .total_cost_usd"'
```

### Custom Test Configuration

```bash
# Create custom test script
cat > test_custom.py << 'EOF'
import asyncio
import httpx

async def test_custom():
    async with httpx.AsyncClient() as client:
        # Start run
        resp = await client.post("http://localhost:8000/runs", json={
            "target_model": "meta-llama/llama-3.1-8b-instruct",
            "max_attempts": 50,
            "cost_cap_usd": 0.5,
            "categories": ["harmful_content"],
            "max_concurrency": 3
        })
        print(f"Started run: {resp.json()}")
        
        # Monitor for 60 seconds
        for _ in range(6):
            await asyncio.sleep(10)
            status = await client.get("http://localhost:8000/status")
            print(f"Progress: {status.json().get('completed_attempts', 0)} attempts")

asyncio.run(test_custom())
EOF

uv run python test_custom.py
```

## Data Access

### View Attempts

```bash
# Show last 10 attempts
tail -n 10 data/attempts.jsonl | jq '.'

# Count total attempts
wc -l data/attempts.jsonl

# Filter by vulnerability status
cat data/attempts.jsonl | jq 'select(.is_vulnerable == true)'

# Group by category
cat data/attempts.jsonl | jq -r '.category' | sort | uniq -c | sort -rn
```

### View Findings

```bash
# Show all findings
cat data/findings.jsonl | jq '.'

# High severity findings only
cat data/findings.jsonl | jq 'select(.severity == "HIGH")'

# Count findings by category
cat data/findings.jsonl | jq -r '.category' | sort | uniq -c
```

### View Reports

```bash
# Find latest report
LATEST_REPORT=$(ls -t data/reports/*.md 2>/dev/null | head -1)

# View report
if [ -n "$LATEST_REPORT" ]; then
    cat "$LATEST_REPORT"
else
    echo "No reports found"
fi

# Generate summary from all reports
find data/reports -name "*.md" -exec grep -H "Total Findings:" {} \;
```

## Database Operations

### Check Run State

```bash
# View current run state
sqlite3 data/state.db "SELECT * FROM state WHERE key='RUN_STATE';"

# View all state keys
sqlite3 data/state.db "SELECT key, json_extract(value, '$.state') as state FROM state;"
```

### View Run History

```bash
# Recent runs
sqlite3 data/state.db "
SELECT 
    id,
    target_model,
    started_at,
    completed_at,
    total_attempts,
    novel_findings
FROM runs 
ORDER BY created_at DESC 
LIMIT 10;"

# Successful runs only
sqlite3 data/state.db "
SELECT * FROM runs 
WHERE status = 'COMPLETED' 
ORDER BY novel_findings DESC;"
```

### Database Maintenance

```bash
# Backup database
cp data/state.db data/state.db.backup.$(date +%Y%m%d)

# Vacuum and optimize
sqlite3 data/state.db "VACUUM; ANALYZE;"

# Check integrity
sqlite3 data/state.db "PRAGMA integrity_check;"

# Reset stuck state (CAUTION!)
sqlite3 data/state.db "UPDATE state SET value=json_set(value, '$.state', 'idle') WHERE key='RUN_STATE';"
```

## Docker Operations

### Build and Deploy

```bash
# Build image
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f runner

# Stop services
docker compose down

# Clean restart with fresh data
docker compose down -v
rm -rf data/*
docker compose up --build -d
```

### Container Management

```bash
# Execute commands in container
docker compose exec runner bash

# Copy files from container
docker compose cp runner:/app/data/findings.jsonl ./findings_backup.jsonl

# View resource usage
docker stats runner

# Restart container
docker compose restart runner
```

## Troubleshooting

### Service Won't Start

```bash
# Check port availability
lsof -i :8000

# Kill process using port
kill -9 $(lsof -t -i:8000)

# Check for Python errors
uv run python -c "from apps.runner.app.main import app; print('Import OK')"

# Verify environment
uv run python -c "import os; print('API Key:', 'SET' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET')"
```

### High Error Rate / 429 Errors

```bash
# Check rate limit status
curl -s http://localhost:8000/status | jq '.error_rate'

# Reduce concurrency
export MAX_CONCURRENCY=2

# Check API key quota
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/auth/key | jq '.'
```

### Cost Overrun Prevention

```bash
# Emergency pause
curl -X POST http://localhost:8000/control/pause

# Check spending report
curl -s http://localhost:8000/status | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Total Cost: \${data.get('total_cost_usd', 0):.4f}\")
print(f\"Cost Cap: \${data.get('cost_cap_usd', 10):.2f}\")
print(f\"Remaining: \${data.get('cost_cap_usd', 10) - data.get('total_cost_usd', 0):.4f}\")
"
```

### Database Issues

```bash
# Fix locked database
fuser -k data/state.db

# Repair WAL mode
sqlite3 data/state.db "PRAGMA journal_mode=WAL; PRAGMA wal_checkpoint(TRUNCATE);"

# Export data before reset
sqlite3 data/state.db ".dump" > backup.sql

# Restore from backup
sqlite3 data/state_new.db < backup.sql
mv data/state_new.db data/state.db
```

### Memory Issues

```bash
# Check memory usage
ps aux | grep python | awk '{sum+=$6} END {print "Total Python Memory: " sum/1024 " MB"}'

# Rotate large files
if [ $(stat -f%z data/attempts.jsonl 2>/dev/null || stat -c%s data/attempts.jsonl) -gt 1073741824 ]; then
    mv data/attempts.jsonl data/attempts.$(date +%Y%m%d).jsonl
    touch data/attempts.jsonl
fi

# Clear cache
find data -name "*.cache" -delete
```

## Performance Tuning

### Optimize Concurrency

```bash
# For high-throughput (more parallel requests)
export MAX_CONCURRENCY=10
export BATCH_SIZE=20
export MAX_CONCURRENT_EVALUATIONS=8

# For cost-efficiency (fewer parallel requests)
export MAX_CONCURRENCY=2
export BATCH_SIZE=5
export MAX_CONCURRENT_EVALUATIONS=2
```

### Adjust Evaluation Pipeline

```yaml
# Edit configs/config.yaml for evaluation tuning
evaluation:
  heuristics_confidence_threshold: 0.9  # Skip judge if very confident
  judge_escalation_threshold: 0.5       # When to use judge
  cache_ttl_hours: 24                   # How long to cache results
  batch_size: 10                        # Attempts per batch
  max_concurrent: 5                     # Parallel evaluations
```

### Bandit Algorithm Tuning

```yaml
# Edit configs/config.yaml
run:
  bandit_algorithm: thompson  # or ucb1
  explore_bias: 0.35         # Higher = more exploration
  epsilon: 0.1               # For epsilon-greedy
  temperature: 1.0           # For softmax selection
```

## Security Best Practices

### API Key Management

```bash
# Never commit keys
echo ".env" >> .gitignore

# Use environment variables
export OPENROUTER_API_KEY="sk-or-..."

# Rotate keys regularly
# Update .env with new key, then restart service
```

### Data Protection

```bash
# Enable PII redaction
export ENABLE_PII_REDACTION=true

# Encrypt sensitive data at rest
# Use disk encryption for data/ directory

# Regular cleanup
find data -name "*.log" -mtime +7 -delete
```

### Access Control

```bash
# Restrict API access (production)
# Use reverse proxy with authentication
# Example nginx config:
cat > nginx.conf << 'EOF'
location /runs {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8000;
}
EOF
```

### Monitoring & Alerts

```bash
# Set up monitoring
export MONITORING_WEBHOOK="https://your-monitoring.com/webhook"
export ALERT_ON_ERROR_RATE=0.2
export ALERT_ON_COST_PERCENT=90

# Log aggregation
tail -f data/*.log | grep -E "ERROR|WARNING" | tee -a alerts.log
```

## Advanced Operations

### Multi-Model Testing

```bash
# Test multiple models in sequence
for model in "meta-llama/llama-3.1-8b-instruct" "meta-llama/llama-3.1-70b-instruct"; do
  echo "Testing $model"
  curl -X POST http://localhost:8000/runs \
    -H "Content-Type: application/json" \
    -d "{\"target_model\": \"$model\", \"max_attempts\": 50}" | jq '.run_id'
  sleep 300  # Wait 5 minutes between runs
done
```

### Continuous Testing Loop

```bash
# Run continuous tests with breaks
while true; do
  # Start run
  RUN_ID=$(curl -s -X POST http://localhost:8000/runs \
    -H "Content-Type: application/json" \
    -d '{"max_attempts": 100, "cost_cap_usd": 1.0}' | jq -r '.run_id')
  
  echo "Started run: $RUN_ID"
  
  # Monitor until complete
  while true; do
    STATE=$(curl -s http://localhost:8000/status | jq -r '.state')
    if [ "$STATE" = "COMPLETED" ] || [ "$STATE" = "idle" ]; then
      break
    fi
    sleep 30
  done
  
  echo "Run $RUN_ID completed"
  sleep 3600  # Wait 1 hour before next run
done
```

## Support & Resources

- **Documentation**: See main [README.md](../README.md)
- **Logs**: Check `data/*.log` for detailed debugging
- **Status Endpoint**: http://localhost:8000/status for real-time metrics
- **Dashboard**: http://localhost:8000/ui for visual monitoring

---

**Remember**: Always monitor costs and set appropriate caps to prevent unexpected charges!