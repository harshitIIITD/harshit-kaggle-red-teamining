#!/usr/bin/env python3
"""
Test script to verify agent integration with mock OpenRouter
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from apps.runner.app.orchestrator import AsyncOrchestrator
from apps.runner.app.store.async_db import AsyncDatabasePool, init_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_mock_integration():
    """Test the orchestrator with mock agents"""
    
    # Set mock API key for testing
    import os
    os.environ["OPENROUTER_API_KEY"] = "mock-api-key-for-testing"
    
    # Test configuration with mock mode enabled
    config = {
        "use_mock_agents": True,  # Enable mock mode for testing
        "run": {
            "max_concurrency": 2,
            "cost_cap_usd": 1.0,
            "categories": ["harmful_content", "prompt_injection"],
        },
        "enabled_categories": ["harmful_content", "prompt_injection"],
        "arms_per_category": 3,
        "planner": {
            "arms_per_category": 3,
            "mutator_complexity": "simple"
        },
        "storage": {
            "database_path": "data/test_state.db",
            "transcripts_path": "data/test_attempts.jsonl"
        },
        "evaluation": {
            "escalation_threshold": 0.7,
            "cache_ttl_seconds": 300
        },
        "target_model": "meta-llama/llama-3.1-8b-instruct"
    }
    
    # Create database pool
    db_pool = AsyncDatabasePool(config["storage"]["database_path"])
    await db_pool.initialize()
    
    # Initialize schema
    async with db_pool.acquire() as conn:
        await init_schema(conn)
    
    try:
        # Create orchestrator
        orchestrator = AsyncOrchestrator(config, db_pool)
        await orchestrator.initialize()
        
        logger.info("Starting mock integration test...")
        
        # Run for a limited number of attempts (match generated tasks)
        result = await orchestrator.run(
            run_id="test-mock-run",
            max_attempts=6  # Only run 6 attempts to match generated tasks
        )
        
        # Print results
        logger.info(f"Test run completed!")
        logger.info(f"  Attempts: {result.get('attempts_count', 0)}")
        logger.info(f"  Successes: {result.get('success_count', 0)}")
        logger.info(f"  Errors: {result.get('error_count', 0)}")
        logger.info(f"  Total cost: ${result.get('total_cost', 0):.4f}")
        
        # Check transcripts were written
        transcript_file = Path(config["storage"]["transcripts_path"])
        if transcript_file.exists():
            with open(transcript_file) as f:
                lines = f.readlines()
                logger.info(f"  Transcripts written: {len(lines)}")
                if lines:
                    import json
                    first_transcript = json.loads(lines[0])
                    logger.info(f"  Sample transcript keys: {list(first_transcript.keys())}")
        
        return result
        
    finally:
        await db_pool.close()


async def test_planner():
    """Test the Planner agent independently"""
    from apps.runner.app.agents.planner import Planner
    
    logger.info("Testing Planner agent...")
    
    planner = Planner(config={
        "arms_per_category": 5,
        "mutator_complexity": "moderate",
        "enabled_categories": ["harmful_content", "prompt_injection", "role_play"]
    })
    
    # Generate backlog
    backlog = planner.generate_backlog(count_per_category=3)
    
    logger.info(f"Generated {len(backlog)} arms")
    
    # Print sample arms
    for i, arm in enumerate(backlog[:5]):
        logger.info(f"  Arm {i}: {arm.category}/{arm.strategy_id} - {arm.template_id}")
        logger.info(f"    Seed: {arm.seed}, Mutators: {arm.mutator_chain}")
        logger.info(f"    Priority: {arm.priority:.2f}")
    
    return backlog


async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting integration tests")
    logger.info("=" * 60)
    
    # Test planner independently
    logger.info("\n1. Testing Planner...")
    await test_planner()
    
    # Test full integration with mock
    logger.info("\n2. Testing full integration with mock agents...")
    await test_mock_integration()
    
    logger.info("\n" + "=" * 60)
    logger.info("Integration tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())