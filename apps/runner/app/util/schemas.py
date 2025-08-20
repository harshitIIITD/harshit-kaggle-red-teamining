# ABOUTME: Pydantic models and TypedDicts for data structures
# ABOUTME: Defines schemas for API responses, database entities, and internal data

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class RunStatus(str, Enum):
    """Run execution states"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Task execution states"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AttemptStatus(str, Enum):
    """Attempt result states"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class SafetyCategory(str, Enum):
    """Safety test categories from spec"""

    # Original categories from config.yaml
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_EXFIL = "system_exfil"
    DECEPTION = "deception"
    SANDBAGGING = "sandbagging"
    SAFETY_VIOLATIONS = "safety_violations"
    PII_LEAKAGE = "pii_leakage"
    COT_MANIP = "cot_manip"
    
    # Additional comprehensive categories
    HARMFUL_CONTENT = "harmful_content"
    MISINFORMATION = "misinformation"
    PRIVACY = "privacy"
    MALICIOUS_USE = "malicious_use"
    MANIPULATION = "manipulation"
    DISCRIMINATION = "discrimination"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SYSTEM_PROMPTS = "system_prompts"
    
    # New enhanced analysis categories
    WHITE_BOX_ANALYSIS = "white_box_analysis"
    BLACK_BOX_ANALYSIS = "black_box_analysis"
    HYBRID_ANALYSIS = "hybrid_analysis"
    ADVANCED_REDTEAM = "advanced_redteam"


class Severity(str, Enum):
    """Issue severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# API Response Schemas
class Usage(BaseModel):
    """Token usage from API response"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class ChatResponse(BaseModel):
    """Structured chat completion response"""

    content: str
    usage: Usage
    model: str
    finish_reason: Optional[str] = None


# Database Entity Schemas
class Run(BaseModel):
    """Run entity for database"""

    id: str
    status: RunStatus
    target_model: str
    config: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_attempts: int = 0
    successful_attempts: int = 0
    total_cost_usd: float = 0.0
    findings_count: int = 0
    error_message: Optional[str] = None


class Task(BaseModel):
    """Task entity for database"""

    id: str
    run_id: str
    status: TaskStatus
    category: SafetyCategory
    strategy: str
    seed: int
    template_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    error_message: Optional[str] = None


class Attempt(BaseModel):
    """Attempt entity for database"""

    id: str
    task_id: str
    run_id: str
    status: AttemptStatus
    prompt: str
    response: Optional[str] = None
    usage: Optional[Usage] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evaluation(BaseModel):
    """Evaluation result for an attempt"""

    attempt_id: str
    is_vulnerable: bool
    category: Optional[SafetyCategory] = None
    severity: Optional[Severity] = None
    confidence: float = Field(ge=0.0, le=1.0)
    heuristic_score: float = 0.0
    judge_score: Optional[float] = None
    rationale: Optional[str] = None
    evaluated_at: datetime


class Finding(BaseModel):
    """Discovered vulnerability finding"""

    id: str
    run_id: str
    cluster_id: str
    category: str  # Changed to str for flexibility
    severity: str  # Changed to str for flexibility  
    confidence: float = 0.0
    novelty_score: float
    best_attempt_id: Optional[str] = None
    example_prompt: Optional[str] = None
    example_response: Optional[str] = None
    prompt: Optional[str] = None  # Added for test compatibility
    response: Optional[str] = None  # Added for test compatibility
    score: float = 0.0  # Added for test compatibility
    evidence: Dict[str, Any] = Field(default_factory=dict)  # Added for test compatibility
    reproduction_steps: List[str] = Field(default_factory=list)
    discovered_at: Optional[datetime] = None
    timestamp: Optional[datetime] = None  # Added for test compatibility
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KaggleFinding(BaseModel):
    """Kaggle competition finding format with self-assessment scores"""
    
    # Core finding identification
    finding_id: str = Field(description="Unique identifier for this finding")
    vulnerability_type: str = Field(description="Type of vulnerability (e.g., jailbreak, prompt_injection)")
    
    # Self-assessment scores (0-10 scale as per Kaggle rubric)
    severity_score: int = Field(ge=0, le=10, description="Severity of harm assessment (0-10)")
    breadth_score: int = Field(ge=0, le=10, description="Breadth of potential impact (0-10)")
    novelty_score: int = Field(ge=0, le=10, description="Novelty of the finding (0-10)")
    reproducibility_score: int = Field(ge=0, le=10, description="Reproducibility assessment (0-10)")
    methodological_insight_score: int = Field(ge=0, le=10, description="Methodological insight value (0-10)")
    
    # Textual descriptions
    title: str = Field(description="Brief descriptive title of the finding")
    description: str = Field(description="Detailed description of the vulnerability")
    impact_assessment: str = Field(description="Assessment of potential real-world impact")
    
    # Reproduction information
    test_prompt: str = Field(description="The prompt that triggers the vulnerability")
    model_response: str = Field(description="The problematic response from the model")
    reproduction_steps: List[str] = Field(description="Step-by-step reproduction instructions")
    
    # Evidence and analysis
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence and analysis")
    mitigation_suggestions: List[str] = Field(default_factory=list, description="Suggested mitigations")
    
    # Metadata
    discovered_at: str = Field(description="ISO timestamp when finding was discovered")
    target_model: str = Field(default="openai/gpt-oss-20b", description="Target model tested")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this finding")
    
    # Competition-specific fields
    competition_submission: bool = Field(default=True, description="Intended for Kaggle submission")
    dataset_license: str = Field(default="CC0", description="Recommended dataset license")


# Request/Response Models
class StartRunRequest(BaseModel):
    """Request to start a new run"""

    target_model: str = "openai/gpt-oss-20b"
    categories: Optional[List[SafetyCategory]] = None
    max_attempts: int = 1000
    cost_cap_usd: float = 10.0
    dry_run: bool = False
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class ControlRequest(BaseModel):
    """Request for control operations"""

    action: Literal["pause", "resume", "stop"]
    reason: Optional[str] = None


class StatusResponse(BaseModel):
    """System status response"""

    status: str
    current_run: Optional[Dict[str, Any]] = None
    counters: Dict[str, int]
    cost_stats: Dict[str, float]
    last_updated: datetime


# Internal Data Structures
class PromptTemplate(BaseModel):
    """Seed prompt template"""

    id: str
    category: SafetyCategory
    template: str
    variables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MutatorChain(BaseModel):
    """Chain of mutations to apply"""

    mutators: List[str]
    seed: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AttemptRecord(BaseModel):
    """Record structure for JSONL attempts file"""
    
    id: str
    timestamp: str
    prompt: str
    model: str
    response: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    cost_usd: float = 0.0
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
