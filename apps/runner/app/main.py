# ABOUTME: Main FastAPI application with health endpoints and control API
# ABOUTME: Provides /status, /ui, /control/* endpoints for system management

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import time
import logging
from pathlib import Path
import asyncio
from dotenv import load_dotenv

from apps.runner.app.util.config import load_config, get_flattened_config
from apps.runner.app.store.async_db import get_db_pool, ensure_schema, get_state, set_state
from apps.runner.app.agents.reporter import Reporter

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    logger.info("Starting red-teaming runner...")
    
    # Initialize app state
    app.state.config = load_config()
    app.state.start_time = time.time()
    app.state.run_state = "idle"  # idle, running, paused, stopped, error
    app.state.counters = {
        "total_attempts": 0,
        "successful_attempts": 0,
        "failed_attempts": 0,
        "findings_count": 0,
        "clusters_count": 0,
        "estimated_cost_usd": 0.0,
    }
    
    # Initialize locks for thread-safe state access
    app.state.counters_lock = asyncio.Lock()
    app.state.run_state_lock = asyncio.Lock()  # Add lock for run_state
    
    logger.info("Configuration loaded successfully")
    
    # Initialize database pool and schema
    db_path = app.state.config["storage"]["sqlite_path"]
    app.state.db_pool = await get_db_pool(db_path)
    await ensure_schema(db_path)
    logger.info("Database pool initialized")

    # Create data directories if they don't exist
    data_dirs = [
        Path(app.state.config["storage"]["sqlite_path"]).parent,
        Path(app.state.config["storage"]["transcripts_path"]).parent,
        Path(app.state.config["storage"]["findings_path"]).parent,
        Path(app.state.config["storage"]["reports_dir"]),
    ]
    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down red-teaming runner...")
    async with app.state.run_state_lock:
        app.state.run_state = "stopped"
    
    # Close database pool
    if hasattr(app.state, 'db_pool') and app.state.db_pool:
        await app.state.db_pool.close()
        logger.info("Database pool closed")
    
    # Close Ollama client if exists
    if hasattr(app.state, 'ollama_client'):
        try:
            await app.state.ollama_client.close()
            logger.info("Ollama client closed")
        except Exception as e:
            logger.error(f"Error closing Ollama client: {e}")


# Create FastAPI app
app = FastAPI(
    title="GPT-OSS-20B Red-Teaming System",
    description="Autonomous vulnerability discovery system for OpenAI Kaggle competition",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Redirect root to UI dashboard"""
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "uptime_seconds": int(time.time() - request.app.state.start_time),
    }


@app.get("/status")
async def status(request: Request) -> Dict[str, Any]:
    """Return detailed system status with counters and configuration"""
    from apps.runner.app.util.format import format_timestamp
    
    # Thread-safe counter access
    async with request.app.state.counters_lock:
        counters_snapshot = request.app.state.counters.copy()
    
    # Calculate error rate
    total = counters_snapshot["total_attempts"]
    failed = counters_snapshot["failed_attempts"]
    error_rate = (failed / total) if total > 0 else 0.0
    
    # Get categories (will be populated from database in production)
    categories = {
        "harmful_content": 0,
        "personal_info": 0,
        "misinformation": 0,
        "intellectual_property": 0,
        "specialized_advice": 0,
        "illegal_activity": 0,
        "hate_harassment": 0,
        "malware": 0,
    }
    
    return {
        "status": request.app.state.run_state,
        "run_state": request.app.state.run_state,  # For backward compatibility
        "total_attempts": counters_snapshot["total_attempts"],
        "successful_attempts": counters_snapshot["successful_attempts"],
        "failed_attempts": counters_snapshot["failed_attempts"],
        "total_cost": counters_snapshot["estimated_cost_usd"],
        "findings_count": counters_snapshot["findings_count"],
        "categories": categories,
        "current_run_id": getattr(request.app.state, "current_run_id", None),  # High Priority Issue #4 fix
        "error_rate": error_rate,
        "timestamp": format_timestamp(),
        "uptime_seconds": int(time.time() - request.app.state.start_time),
        "counters": counters_snapshot,  # For backward compatibility
        "config": get_flattened_config(request.app.state.config),  # For backward compatibility
    }


@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """Serve minimal dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Red-Teaming Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="30">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }
            .status-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric {
                display: inline-block;
                margin: 10px 20px 10px 0;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }
            .control-buttons {
                margin: 20px 0;
            }
            button {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                margin: 0 5px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1em;
            }
            button:hover {
                background: #0056b3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .state-idle { color: #666; }
            .state-running { color: #28a745; }
            .state-paused { color: #ffc107; }
            .state-stopped { color: #dc3545; }
            .state-error { color: #dc3545; }
        </style>
    </head>
    <body>
        <h1>🎯 GPT-OSS-20B Red-Teaming Dashboard</h1>
        
        <div class="status-card">
            <h2>System Status</h2>
            <div id="status-container">
                <div class="metric">
                    <div class="metric-label">Run State</div>
                    <div class="metric-value" id="run-state">Loading...</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Attempts</div>
                    <div class="metric-value" id="total-attempts">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Successful</div>
                    <div class="metric-value" id="successful-attempts">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Failed</div>
                    <div class="metric-value" id="failed-attempts">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Findings</div>
                    <div class="metric-value" id="findings">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Clusters</div>
                    <div class="metric-value" id="clusters">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Cost</div>
                    <div class="metric-value" id="cost">$-</div>
                </div>
            </div>
        </div>
        
        <div class="status-card">
            <h2>Controls</h2>
            <div class="control-buttons">
                <button id="btn-pause" onclick="controlAction('pause')">⏸️ Pause</button>
                <button id="btn-resume" onclick="controlAction('resume')">▶️ Resume</button>
                <button id="btn-stop" onclick="controlAction('stop')">⏹️ Stop</button>
            </div>
        </div>
        
        <div class="status-card">
            <h2>Configuration</h2>
            <div id="config-container">
                <div class="metric">
                    <div class="metric-label">Target Model</div>
                    <div class="metric-value" id="target-model" style="font-size: 1em;">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Concurrency</div>
                    <div class="metric-value" id="max-concurrency">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Cost Cap</div>
                    <div class="metric-value" id="cost-cap">$-</div>
                </div>
            </div>
        </div>
        
        <div class="status-card">
            <h2>Reports</h2>
            <div id="reports-container">
                <p>Report will be available once findings are discovered.</p>
                <div class="metric">
                    <div class="metric-label">Latest Report</div>
                    <div class="metric-value" id="latest-report" style="font-size: 1em;">
                        <a href="#" id="report-link">No report available</a>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function updateStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    // Update run state
                    const stateElem = document.getElementById('run-state');
                    stateElem.textContent = data.run_state.toUpperCase();
                    stateElem.className = 'metric-value state-' + data.run_state;
                    
                    // Update counters
                    document.getElementById('total-attempts').textContent = data.counters.total_attempts;
                    document.getElementById('successful-attempts').textContent = data.counters.successful_attempts;
                    document.getElementById('failed-attempts').textContent = data.counters.failed_attempts;
                    document.getElementById('findings').textContent = data.counters.findings_count;
                    document.getElementById('clusters').textContent = data.counters.clusters_count;
                    document.getElementById('cost').textContent = '$' + data.counters.estimated_cost_usd.toFixed(2);
                    
                    // Update config
                    document.getElementById('target-model').textContent = data.config.target_model;
                    document.getElementById('max-concurrency').textContent = data.config.max_concurrency;
                    document.getElementById('cost-cap').textContent = '$' + data.config.cost_cap_usd;
                    
                    // Update button states
                    document.getElementById('btn-pause').disabled = data.run_state !== 'running';
                    document.getElementById('btn-resume').disabled = data.run_state !== 'paused';
                    document.getElementById('btn-stop').disabled = data.run_state === 'stopped' || data.run_state === 'idle';
                } catch (error) {
                    console.error('Failed to update status:', error);
                }
            }
            
            async function controlAction(action) {
                try {
                    const response = await fetch('/control/' + action, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'}
                    });
                    const data = await response.json();
                    console.log(action + ' response:', data);
                    updateStatus();
                } catch (error) {
                    console.error('Control action failed:', error);
                }
            }
            
            // Update status every 3 seconds
            updateStatus();
            setInterval(updateStatus, 3000);
        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/control/pause")
async def control_pause(request: Request) -> Dict[str, str]:
    """Pause the current run"""
    pool = request.app.state.db_pool
    
    async with pool.acquire() as conn:
        # Get current state from database
        current_state = await get_state(conn, "RUN_STATE") or "idle"
        
        if current_state == "running":
            # Update database state
            await set_state(conn, "RUN_STATE", "paused")
            
            # Update app state with lock
            async with request.app.state.run_state_lock:
                request.app.state.run_state = "paused"
            
            logger.info("Run paused")
            return {"status": "paused", "message": "Run has been paused"}
        elif current_state == "paused":
            return {"status": "already_paused", "message": "Run is already paused"}
        else:
            return {
                "status": "not_running",
                "message": f"Cannot pause from state: {current_state}",
            }


@app.post("/control/resume")
async def control_resume(request: Request) -> Dict[str, str]:
    """Resume a paused run"""
    pool = request.app.state.db_pool
    
    async with pool.acquire() as conn:
        # Get current state from database
        current_state = await get_state(conn, "RUN_STATE") or "idle"
        
        if current_state == "paused":
            # Update database state
            await set_state(conn, "RUN_STATE", "running")
            
            # Update app state with lock
            async with request.app.state.run_state_lock:
                request.app.state.run_state = "running"
            
            logger.info("Run resumed")
            return {"status": "resumed", "message": "Run has been resumed"}
        elif current_state == "running":
            return {"status": "already_running", "message": "Run is already running"}
        else:
            return {
                "status": "cannot_resume",
                "message": f"Cannot resume from state: {current_state}",
            }


@app.post("/control/stop")
async def control_stop(request: Request) -> Dict[str, str]:
    """Stop the current run"""
    pool = request.app.state.db_pool
    
    async with pool.acquire() as conn:
        # Get current state from database
        current_state = await get_state(conn, "RUN_STATE") or "idle"
        
        if current_state in ["running", "paused"]:
            # Update database state
            await set_state(conn, "RUN_STATE", "stopped")
            
            # Update app state with lock
            async with request.app.state.run_state_lock:
                request.app.state.run_state = "stopped"
            
            logger.info(f"Run stopped from state: {current_state}")
            return {"status": "stopped", "message": "Run has been stopped"}
        elif current_state == "stopped":
            return {"status": "already_stopped", "message": "Run is already stopped"}
        else:
            # Force stop from any state
            await set_state(conn, "RUN_STATE", "stopped")
            
            async with request.app.state.run_state_lock:
                request.app.state.run_state = "stopped"
            
            return {
                "status": "stopped",
                "message": f"Run state set to stopped from: {current_state}",
            }


@app.post("/runs")
async def start_run(request: Request, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Start a new red-teaming run with orchestrator"""
    import uuid
    from apps.runner.app.agents.reporter import Reporter
    from apps.runner.app.agents.novelty import NoveltyDetector
    
    # Check if a run is already in progress (with lock)
    async with request.app.state.run_state_lock:
        if request.app.state.run_state == "running":
            raise HTTPException(status_code=400, detail="A run is already in progress")
        
        # Parse parameters
        params = params or {}
        run_id = params.get("run_id", f"run-{uuid.uuid4().hex[:8]}")
        target_model = params.get("target_model", request.app.state.config["run"]["target_model"])
        max_attempts = params.get("max_attempts", 200)  # Default for dry-run
        cost_cap = params.get("cost_cap_usd", request.app.state.config["run"]["cost_cap_usd"])
        dry_run = params.get("dry_run", False)
        
        # Override model for dry-run
        if dry_run:
            target_model = "meta-llama/llama-3.1-8b-instruct"
            logger.info(f"Dry-run mode: using {target_model}")
        
        # Update config for this run - use deepcopy to avoid mutating global config
        import copy
        run_config = copy.deepcopy(request.app.state.config)
        run_config["run"]["target_model"] = target_model
        run_config["run"]["cost_cap_usd"] = cost_cap
        run_config["run"]["max_attempts"] = max_attempts
        
        # Set run state (already under lock)
        request.app.state.run_state = "running"
        request.app.state.current_run_id = run_id
    
    # Update database state using async pool
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        await set_state(conn, "RUN_STATE", "running")
        await set_state(conn, "CURRENT_RUN_ID", run_id)
    
    logger.info(f"Starting run {run_id} with model {target_model}, cost cap ${cost_cap}")
    
    # Start orchestrator in background
    async def run_orchestrator(app):
        try:
            # Use async database pool for orchestrator
            from apps.runner.app.orchestrator import AsyncOrchestrator
            orchestrator = AsyncOrchestrator(run_config, app.state.db_pool)
            detector = NoveltyDetector(run_config)
            reporter = Reporter(
                findings_file=run_config["storage"]["findings_path"],
                reports_dir=run_config["storage"]["reports_dir"],
            )
            
            # Run orchestrator
            result = await orchestrator.run(
                run_id=run_id,
                max_attempts=max_attempts,
                detector=detector,
                reporter=reporter,
            )
            
            # Update counters with thread-safe lock
            async with app.state.counters_lock:
                app.state.counters["total_attempts"] = result.get("total_attempts", 0)
                app.state.counters["successful_attempts"] = result.get("successful_attempts", 0)
                app.state.counters["failed_attempts"] = result.get("failed_attempts", 0)
                app.state.counters["findings_count"] = result.get("findings_count", 0)
                app.state.counters["clusters_count"] = result.get("clusters_count", 0)
                app.state.counters["estimated_cost_usd"] = result.get("total_cost", 0.0)
            
            # Log report path if generated (report is already generated in orchestrator)
            if "report_path" in result:
                logger.info(f"Report available at: {result['report_path']}")
            
            # Update state to completed using async pool
            async with app.state.run_state_lock:
                app.state.run_state = "completed"
            
            pool = app.state.db_pool
            async with pool.acquire() as conn:
                await set_state(conn, "RUN_STATE", "completed")
            
            logger.info(f"Run {run_id} completed successfully")
            return result
            
        except Exception as e:
            logger.exception(f"Run {run_id} failed with error")
            
            # Update state with lock
            async with app.state.run_state_lock:
                app.state.run_state = "error"
            
            # Update database using async pool
            try:
                pool = app.state.db_pool
                async with pool.acquire() as conn:
                    await set_state(conn, "RUN_STATE", "error")
            except Exception as db_error:
                logger.error(f"Failed to update database on error: {db_error}")
            
            # Don't re-raise in background task - just log and return
            return {"status": "error", "run_id": run_id, "error": str(e)}
    
    # Start run in background task
    asyncio.create_task(run_orchestrator(request.app))
    
    return {
        "status": "started",
        "run_id": run_id,
        "message": f"Run started with model {target_model}",
        "params": {
            "target_model": target_model,
            "max_attempts": max_attempts,
            "cost_cap_usd": cost_cap,
            "dry_run": dry_run,
        },
    }
