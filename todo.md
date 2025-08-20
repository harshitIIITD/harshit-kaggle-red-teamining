
# todo.md — Red‑Teaming Runner

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked

## Spikes
- [ ] OR‑001 Concurrency & 429 profile — Owner: ENG — Due: D2 AM — Kill: error rate >20% for 3m
- [ ] OR‑002 Model IDs & pricing — Owner: ENG — Due: D1 AM
- [ ] CT‑001 Token/cost envelope — Owner: ENG — Due: D1 PM
- [ ] PR‑001 Pause/resume semantics — Owner: ENG — Due: D2 AM
- [ ] EV‑001 Judge calibration — Owner: ENG — Due: D2 AM
- [ ] NV‑001 MinHash/Jaccard params — Owner: ENG — Due: D2 PM
- [ ] RP‑001 Reporter/redaction — Owner: ENG — Due: D2 PM
- [ ] UI‑001 UI health card — Owner: ENG — Due: D2 PM

## Vertical Slices
### Slice A — Hello Runner & Health
- [x] Project skeleton with FastAPI, config loader, tests
- [x] OpenRouter client with retries & cost meter, tests

### Slice B — Seed+Mutator → JSONL
- [x] SQLite schema & DAO with WAL + migrations (Step 3 COMPLETED)
- [x] Seed template registry + mutators (deterministic), property tests (Step 4 COMPLETED)
- [x] Attempt runner appends sanitized JSONL; CLI (Step 5 COMPLETED)

### Slice C — Evaluate & Dedupe
- [x] Heuristics evaluator; judge integration; threshold knob (Step 6 COMPLETED)
- [x] Novelty MinHash + cluster + promotion (Step 7 COMPLETED)

### Slice D — Orchestrate & Operate
- [x] Bandit orchestrator; bounded concurrency; checkpointing (Step 8 COMPLETED)
- [x] Pause/resume semantics; crash recovery; circuit breaker (Step 9 COMPLETED)
- [x] Cost cap enforcement (Step 9 COMPLETED)

### Slice E — Observe & Report
- [x] Minimal /ui dashboard; counters, spend, last findings (Step 10 COMPLETED)
- [x] Rolling Markdown reporter with redaction (Step 11 COMPLETED)
- [x] E2E dry‑run on Llama‑8B; short target run on gpt‑oss‑20b (Step 12 COMPLETED)

## Final Integration
- [x] Docker Compose setup with volume mounts (COMPLETED)
- [x] Multi-stage Dockerfile with uv (COMPLETED)
- [x] Smoke tests for all endpoints (COMPLETED)
- [x] Fixed async database test issues (COMPLETED)
- [x] E2E test infrastructure for 100+ attempts (COMPLETED 2025-08-10)
  - Created test scripts for API-based testing
  - Verified pipeline works with Llama-8B model
  - Successfully processes tasks and evaluations

## Runbooks
- [x] curl scripts for /runs, /control/pause, /control/resume (COMPLETED)
- [x] Troubleshooting guide (COMPLETED)

## Follow-ups from Real API Integration (2025-08-09)

### Completed Today (2025-08-10)
- [x] Created comprehensive cost monitoring system
  - Implemented CostTracker with configurable alert thresholds
  - Added CostAlerter for webhook/email notifications
  - Created spending reports and metrics
  - Added checkpoint persistence for crash recovery
  - Wrote comprehensive test suite (13 tests)
- [x] Performed code review with o3 model
  - Identified critical timing bug and async I/O issues
  - Found thread safety concerns
  - Documented fixes needed for production readiness

### Completed on 2025-08-09
- [x] Fixed hanging async database pool tests (test_pool_exhaustion_handling)
  - Fixed semaphore blocking issue in connection acquisition
- [x] Enhanced OpenRouterClient retry logic for 429 errors
  - Added custom RateLimitError exception
  - Implemented wait_rate_limit strategy that respects retry-after headers
  - Added better error messages and logging
- [x] Wired up Reporter in orchestrator
  - Reports now automatically generated after run completion
  - Integrated with NoveltyDetector for findings
- [x] Validated seed templates format
  - All 30 templates loading correctly across 10 categories
  - PromptCrafter successfully using templates with mutators

## Follow-ups from Real API Integration

### Critical Fixes
- [x] Fix Attempt schema validation errors in evaluator
  - Missing fields: run_id, status, started_at
  - Blocking full evaluation pipeline
- [x] Fix OpenRouterClient initialization in orchestrator
  - Currently ignoring config parameters
  - Should parse timeout, base_url from config

### Agent Improvements
- [x] Implement real template loading in PromptCrafter
  - Currently falls back to simple prompts
  - Need actual attack templates in seeds/ directory
- [x] Complete Evaluator integration
  - Wire up heuristics → judge escalation
  - Implement score calculation and vulnerability detection
- [x] Add NoveltyDetector integration
  - Currently not being used in orchestrator
  - Need MinHash/Jaccard clustering for deduplication
- [x] Implement Reporter agent
  - Generate Markdown reports from findings
  - Add PII redaction

### Testing & Validation
- [x] Add retry logic for transient API errors
  - Handle 429 rate limits gracefully
  - Implement exponential backoff
- [x] Create comprehensive integration tests (COMPLETED 2025-08-09)
  - Test full pipeline with mock API
  - Validate all agent interactions
- [x] Add cost monitoring and alerts (COMPLETED 2025-08-09)
  - Track spending per run
  - Alert when approaching cost cap
  - Implemented CostTracker with thresholds
  - Created CostAlerter for webhook/email notifications
  - Added spending reports and metrics
  - Code reviewed with o3 model

### Code Review Fixes (from o3 review 2025-08-09)
#### Critical Fixes
- [x] Fix timing bug in CostTracker (line 135: use .total_seconds() not .seconds) (COMPLETED 2025-08-10)
- [x] Add asyncio.Lock for thread-safe cost additions (COMPLETED 2025-08-10)
- [x] Replace sync file I/O with aiofiles for async operations (COMPLETED 2025-08-10)

#### High Priority
- [x] Add input validation for costs (no negative values) (COMPLETED 2025-08-10)
- [x] Validate webhook URLs before making requests (COMPLETED 2025-08-10)
- [x] Add error handling for file permission issues (COMPLETED 2025-08-10)

#### Medium Priority  
- [x] Make log file path configurable (currently hardcoded) (COMPLETED 2025-08-10)
- [x] Optimize request_costs list cleanup (currently O(n)) (COMPLETED 2025-08-10)
- [x] Add path sanitization in set_checkpoint_file() (COMPLETED 2025-08-10)

### Performance & Scale
- [ ] Optimize for larger test runs
  - Test with 100+ attempts
  - Tune concurrency settings
- [x] Add batch processing for evaluations (COMPLETED 2025-08-10)
  - Implemented batch evaluation methods with concurrency control
  - Added intelligent grouping by category for cache efficiency
  - Created sub-batch processing with semaphore rate limiting
  - Added optimized batch evaluation with category grouping
- [ ] Implement progressive task generation
  - Don't generate all tasks upfront
  - Use bandit feedback to guide generation

### Configuration & Operations
- [ ] Create seed attack templates
  - Populate seeds/ directory with real templates
  - Cover all 8 attack categories
- [ ] Add configuration validation
  - Validate model IDs exist
  - Check API key permissions
- [ ] Implement graceful shutdown
  - Save state on SIGTERM
  - Resume from last checkpoint
- [ ] Add metrics and monitoring
  - Export Prometheus metrics
  - Track success rates by category

### Documentation
- [ ] Document API response formats
  - Create examples of each attack type
  - Show expected model responses
- [ ] Write deployment guide
  - Production configuration
  - Security considerations
- [ ] Create attack template authoring guide
  - Template format specification
  - Best practices for mutations
