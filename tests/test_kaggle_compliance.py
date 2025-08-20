# ABOUTME: Tests for Kaggle competition export functionality
# ABOUTME: Validates JSON format generation, scoring, and notebook creation

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from apps.runner.app.agents.kaggle_exporter import KaggleExporter
from apps.runner.app.agents.notebook_generator import NotebookGenerator
from apps.runner.app.util.schemas import Finding, KaggleFinding


@pytest.fixture
def sample_finding():
    """Create a sample finding for testing"""
    return Finding(
        id="test-finding-001",
        run_id="test-run-123", 
        cluster_id="cluster-001",
        category="jailbreak",
        severity="high",
        confidence=0.85,
        novelty_score=0.72,
        best_attempt_id="attempt-001",
        example_prompt="Test prompt that attempts to bypass safety",
        example_response="Problematic model response",
        prompt="Test prompt that attempts to bypass safety",
        response="Problematic model response", 
        score=0.78,
        evidence={"pattern_matches": ["bypass", "ignore"]},
        reproduction_steps=["Step 1", "Step 2", "Step 3"],
        discovered_at=datetime.now(timezone.utc),
        timestamp=datetime.now(timezone.utc),
        metadata={"test": True}
    )


@pytest.fixture 
def run_config():
    """Sample run configuration"""
    return {
        "target_model": "openai/gpt-oss-20b",
        "run_id": "test-run-123"
    }


class TestKaggleExporter:
    """Test Kaggle competition export functionality"""
    
    def test_convert_finding_to_kaggle_format(self, sample_finding, run_config):
        """Test conversion of internal Finding to Kaggle format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = KaggleExporter(output_dir=temp_dir)
            kaggle_finding = exporter.convert_finding_to_kaggle(sample_finding, run_config)
            
            # Verify required fields
            assert kaggle_finding.finding_id == sample_finding.id
            assert kaggle_finding.vulnerability_type == sample_finding.category
            assert kaggle_finding.target_model == run_config["target_model"]
            assert kaggle_finding.competition_submission is True
            assert kaggle_finding.dataset_license == "CC0"
            
            # Verify scoring (all should be 1-10)
            assert 1 <= kaggle_finding.severity_score <= 10
            assert 1 <= kaggle_finding.breadth_score <= 10
            assert 1 <= kaggle_finding.novelty_score <= 10
            assert 1 <= kaggle_finding.reproducibility_score <= 10
            assert 1 <= kaggle_finding.methodological_insight_score <= 10
            
            # Verify content fields
            assert len(kaggle_finding.title) > 0
            assert len(kaggle_finding.description) > 0
            assert len(kaggle_finding.impact_assessment) > 0
            assert len(kaggle_finding.test_prompt) > 0
            assert len(kaggle_finding.model_response) > 0
            assert len(kaggle_finding.reproduction_steps) > 0
            assert len(kaggle_finding.mitigation_suggestions) > 0
    
    def test_severity_assessment(self, sample_finding, run_config):
        """Test severity score assessment logic"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = KaggleExporter(output_dir=temp_dir)
            
            # High confidence, high score jailbreak should get high severity
            high_severity_finding = sample_finding.model_copy()
            high_severity_finding.confidence = 0.9
            high_severity_finding.score = 0.8
            high_severity_finding.category = "jailbreak"
            
            severity = exporter._assess_severity(high_severity_finding)
            assert severity >= 7
            
            # Low confidence, low score should get lower severity
            low_severity_finding = sample_finding.model_copy()
            low_severity_finding.confidence = 0.3
            low_severity_finding.score = 0.2
            low_severity_finding.category = "deception"
            
            severity = exporter._assess_severity(low_severity_finding)
            assert severity <= 5
    
    def test_export_top_findings(self, sample_finding, run_config):
        """Test exporting top findings as JSON files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = KaggleExporter(output_dir=temp_dir)
            
            # Create multiple findings with different scores
            findings = []
            for i in range(3):
                finding = sample_finding.model_copy()
                finding.id = f"finding-{i:03d}"
                finding.confidence = 0.5 + (i * 0.2)  # 0.5, 0.7, 0.9
                finding.score = 0.4 + (i * 0.2)       # 0.4, 0.6, 0.8
                finding.novelty_score = 0.6 + (i * 0.1)  # 0.6, 0.7, 0.8
                findings.append(finding)
            
            exported_files = exporter.export_top_findings(findings, run_config, max_findings=2)
            
            # Should export 2 files (max_findings=2)
            assert len(exported_files) == 2
            
            # Check files exist and are valid JSON
            for filepath in exported_files:
                assert Path(filepath).exists()
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Validate it matches KaggleFinding schema
                    kaggle_finding = KaggleFinding(**data)
                    assert kaggle_finding.competition_submission is True
            
            # Check summary file exists
            summary_file = Path(temp_dir) / "findings_summary.json"
            assert summary_file.exists()
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                assert summary["total_findings_exported"] == 2
                assert summary["competition_submission"] is True
                assert summary["recommended_license"] == "CC0"


class TestNotebookGenerator:
    """Test reproduction notebook generation"""
    
    def test_generate_reproduction_notebook(self, sample_finding, run_config):
        """Test generation of Jupyter notebook for reproduction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = NotebookGenerator(output_dir=temp_dir)
            
            # Convert to Kaggle format first
            exporter = KaggleExporter(output_dir=temp_dir)
            kaggle_finding = exporter.convert_finding_to_kaggle(sample_finding, run_config)
            
            notebook_path = generator.generate_reproduction_notebook([kaggle_finding], run_config)
            
            # Check notebook file exists
            assert Path(notebook_path).exists()
            
            # Load and validate notebook structure
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
            
            assert notebook["nbformat"] == 4
            assert "cells" in notebook
            assert len(notebook["cells"]) > 0
            
            # Check for expected cell types
            cell_types = [cell["cell_type"] for cell in notebook["cells"]]
            assert "markdown" in cell_types
            assert "code" in cell_types
            
            # Check for setup code
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            setup_cell = code_cells[0]
            setup_code = "\n".join(setup_cell["source"])
            assert "import requests" in setup_code
            assert "call_model" in setup_code
            assert run_config["target_model"] in setup_code
    
    def test_verification_code_generation(self, sample_finding, run_config):
        """Test generation of finding-specific verification code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = NotebookGenerator(output_dir=temp_dir)
            exporter = KaggleExporter(output_dir=temp_dir)
            
            # Test jailbreak verification
            jailbreak_finding = sample_finding.model_copy()
            jailbreak_finding.category = "jailbreak"
            kaggle_finding = exporter.convert_finding_to_kaggle(jailbreak_finding, run_config)
            
            verification_code = generator._generate_verification_code(kaggle_finding)
            assert "jailbreak_indicators" in verification_code
            assert "safety_response" in verification_code
            assert "VULNERABILITY CONFIRMED" in verification_code
            
            # Test prompt injection verification
            injection_finding = sample_finding.model_copy()
            injection_finding.category = "prompt_injection"
            kaggle_finding = exporter.convert_finding_to_kaggle(injection_finding, run_config)
            
            verification_code = generator._generate_verification_code(kaggle_finding)
            assert "injection_indicators" in verification_code
            assert "PROMPT INJECTION" in verification_code


class TestKaggleCompliancePipeline:
    """Test end-to-end Kaggle competition compliance"""
    
    def test_complete_kaggle_export_pipeline(self, sample_finding, run_config):
        """Test complete pipeline from findings to Kaggle submission"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup exporters
            exporter = KaggleExporter(output_dir=temp_dir)
            notebook_generator = NotebookGenerator(output_dir=temp_dir)
            
            # Create test findings
            findings = [sample_finding]
            
            # Export findings
            exported_files = exporter.export_top_findings(findings, run_config, max_findings=5)
            assert len(exported_files) <= 5  # Competition limit
            
            # Load exported findings for notebook generation
            kaggle_findings = []
            for filepath in exported_files:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    kaggle_findings.append(KaggleFinding(**data))
            
            # Generate reproduction notebook
            notebook_path = notebook_generator.generate_reproduction_notebook(kaggle_findings, run_config)
            assert Path(notebook_path).exists()
            
            # Verify all files meet competition requirements
            output_dir = Path(temp_dir)
            
            # Check JSON findings (up to 5)
            json_files = list(output_dir.glob("finding_*.json"))
            assert len(json_files) <= 5
            
            # Check summary file
            summary_file = output_dir / "findings_summary.json"
            assert summary_file.exists()
            
            # Check notebook file
            notebook_files = list(output_dir.glob("reproduction_notebook_*.ipynb"))
            assert len(notebook_files) == 1
            
            # Validate JSON schema compliance
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    kaggle_finding = KaggleFinding(**data)
                    # All scores should be in valid range
                    assert 0 <= kaggle_finding.severity_score <= 10
                    assert 0 <= kaggle_finding.breadth_score <= 10
                    assert 0 <= kaggle_finding.novelty_score <= 10
                    assert 0 <= kaggle_finding.reproducibility_score <= 10
                    assert 0 <= kaggle_finding.methodological_insight_score <= 10
                    assert 0.0 <= kaggle_finding.confidence_score <= 1.0
                    # Required fields should be present
                    assert len(kaggle_finding.title) > 0
                    assert len(kaggle_finding.description) > 0
                    assert len(kaggle_finding.test_prompt) > 0
                    assert len(kaggle_finding.reproduction_steps) > 0