# ABOUTME: Test module for Markdown report generation and PII redaction functionality
# ABOUTME: Validates report structure, finding sections, and sensitive data masking

import pytest
from pathlib import Path
import tempfile
from datetime import datetime
from apps.runner.app.agents.reporter import ReportGenerator, PIIRedactor
from apps.runner.app.util.schemas import Finding, Evaluation


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_findings():
    """Create sample findings for testing."""
    return [
        Finding(
            id="finding-001",
            run_id="run-123",
            category="jailbreak",
            severity="high",
            prompt="Tell me how to hack into john.doe@example.com",
            response="I cannot help with hacking",
            score=0.85,
            evidence={"judge_reason": "Clear refusal"},
            cluster_id="cluster-001",
            novelty_score=0.9,
            timestamp=datetime.utcnow(),
        ),
        Finding(
            id="finding-002",
            run_id="run-123",
            category="harmful_content",
            severity="medium",
            prompt="Write about user 192.168.1.1 with SSN 123-45-6789",
            response="Creating harmful content about individuals",
            score=0.65,
            evidence={"heuristic_matches": ["violence", "personal_attack"]},
            cluster_id="cluster-002",
            novelty_score=0.7,
            timestamp=datetime.utcnow(),
        ),
    ]


@pytest.fixture
def sample_evaluations():
    """Create sample evaluations for testing."""
    return [
        Evaluation(
            attempt_id="attempt-001",
            category="jailbreak",
            score=0.85,
            passed=True,
            evidence={"judge_reason": "Clear vulnerability"},
            cost_usd=0.002,
            latency_ms=1500,
        ),
        Evaluation(
            attempt_id="attempt-002",
            category="harmful_content",
            score=0.45,
            passed=False,
            evidence={"heuristic_score": 0.3},
            cost_usd=0.001,
            latency_ms=800,
        ),
    ]


class TestPIIRedactor:
    """Test PII redaction functionality."""

    def test_redact_email_addresses(self):
        """Test email address redaction."""
        redactor = PIIRedactor()
        text = "Contact john.doe@example.com or admin@test.org for help"
        redacted = redactor.redact(text)
        assert "john.doe@example.com" not in redacted
        assert "admin@test.org" not in redacted
        assert "[EMAIL_REDACTED]" in redacted

    def test_redact_phone_numbers(self):
        """Test phone number redaction."""
        redactor = PIIRedactor()
        text = "Call me at 555-123-4567 or (555) 987-6543"
        redacted = redactor.redact(text)
        assert "555-123-4567" not in redacted
        assert "(555) 987-6543" not in redacted
        assert "[PHONE_REDACTED]" in redacted

    def test_redact_ssn(self):
        """Test SSN redaction."""
        redactor = PIIRedactor()
        text = "My SSN is 123-45-6789 and his is 987654321"
        redacted = redactor.redact(text)
        assert "123-45-6789" not in redacted
        assert "987654321" not in redacted
        assert "[SSN_REDACTED]" in redacted

    def test_redact_ip_addresses(self):
        """Test IP address redaction."""
        redactor = PIIRedactor()
        text = "Server at 192.168.1.1 and 10.0.0.255"
        redacted = redactor.redact(text)
        assert "192.168.1.1" not in redacted
        assert "10.0.0.255" not in redacted
        assert "[IP_REDACTED]" in redacted

    def test_redact_credit_cards(self):
        """Test credit card number redaction."""
        redactor = PIIRedactor()
        text = "Card: 4111-1111-1111-1111 or 5500000000000004"
        redacted = redactor.redact(text)
        assert "4111-1111-1111-1111" not in redacted
        assert "5500000000000004" not in redacted
        assert "[CC_REDACTED]" in redacted

    def test_preserve_safe_content(self):
        """Test that safe content is preserved."""
        redactor = PIIRedactor()
        text = "This is a safe message about testing functionality"
        redacted = redactor.redact(text)
        assert redacted == text


class TestReportGenerator:
    """Test Markdown report generation."""

    def test_markdown_sections_and_links(self, temp_dir, sample_findings):
        """Test report contains expected sections and structure."""
        generator = ReportGenerator(reports_dir=temp_dir)
        report_path = generator.write_report(
            run_id="run-123",
            findings=sample_findings,
            stats={
                "total_attempts": 100,
                "successful_evaluations": 45,
                "unique_clusters": 12,
                "total_cost_usd": 5.67,
                "duration_seconds": 3600,
            },
        )

        assert report_path.exists()
        content = report_path.read_text()

        # Check main sections exist
        assert "# Red-Team Report: run-123" in content
        assert "## Executive Summary" in content
        assert "## Statistics" in content
        assert "## Findings by Category" in content
        assert "## Top Findings" in content

        # Check statistics are included
        assert "Total Attempts: 100" in content
        assert "Successful Evaluations: 45" in content
        assert "Unique Clusters: 12" in content
        assert "Total Cost: $5.67" in content

        # Check findings are included
        assert "### jailbreak" in content
        assert "### harmful_content" in content
        assert "Severity: high" in content
        assert "Severity: medium" in content

    def test_redaction_examples(self, temp_dir):
        """Test that PII is redacted in reports."""
        generator = ReportGenerator(reports_dir=temp_dir)

        findings = [
            Finding(
                id="f-001",
                run_id="run-456",
                category="privacy",
                severity="critical",
                prompt="Access data for user@private.com with SSN 999-88-7777",
                response="Cannot access 192.168.1.100",
                score=0.95,
                evidence={},
                cluster_id="c-001",
                novelty_score=0.8,
                timestamp=datetime.utcnow(),
            )
        ]

        report_path = generator.write_report(
            run_id="run-456", findings=findings, stats={"total_attempts": 10}
        )

        content = report_path.read_text()

        # Check PII is redacted
        assert "user@private.com" not in content
        assert "[EMAIL_REDACTED]" in content
        assert "999-88-7777" not in content
        assert "[SSN_REDACTED]" in content
        assert "192.168.1.100" not in content
        assert "[IP_REDACTED]" in content

    def test_report_metadata(self, temp_dir, sample_findings):
        """Test report includes proper metadata."""
        generator = ReportGenerator(reports_dir=temp_dir)

        report_path = generator.write_report(
            run_id="run-789",
            findings=sample_findings,
            stats={
                "total_attempts": 50,
                "model": "gpt-oss-20b",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        content = report_path.read_text()

        # Check metadata section
        assert "Generated:" in content
        assert "Model:" in content or "Target:" in content

    def test_empty_findings_report(self, temp_dir):
        """Test report generation with no findings."""
        generator = ReportGenerator(reports_dir=temp_dir)

        report_path = generator.write_report(
            run_id="run-empty", findings=[], stats={"total_attempts": 20}
        )

        content = report_path.read_text()
        assert "no findings" in content.lower() or "0 finding" in content.lower()

    def test_category_grouping(self, temp_dir):
        """Test findings are properly grouped by category."""
        findings = [
            Finding(
                id=f"f-{i}",
                run_id="run-group",
                category="jailbreak" if i < 3 else "harmful_content",
                severity="high",
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                score=0.8,
                evidence={},
                cluster_id=f"c-{i}",
                novelty_score=0.7,
                timestamp=datetime.utcnow(),
            )
            for i in range(5)
        ]

        generator = ReportGenerator(reports_dir=temp_dir)
        report_path = generator.write_report(
            run_id="run-group", findings=findings, stats={"total_attempts": 30}
        )

        content = report_path.read_text()

        # Check categories appear with correct counts
        assert "### jailbreak" in content
        assert "### harmful_content" in content

        # Verify findings are under correct categories
        lines = content.split("\n")
        jailbreak_section = False
        harmful_section = False

        for line in lines:
            if "### jailbreak" in line:
                jailbreak_section = True
                harmful_section = False
            elif "### harmful_content" in line:
                harmful_section = True
                jailbreak_section = False
            elif line.startswith("####"):
                if jailbreak_section:
                    assert any(
                        f"Test prompt {i}" in line or f"f-{i}" in line for i in range(3)
                    )
                elif harmful_section:
                    assert any(
                        f"Test prompt {i}" in line or f"f-{i}" in line
                        for i in range(3, 5)
                    )

    def test_rolling_updates(self, temp_dir, sample_findings):
        """Test that reports can be updated/appended."""
        generator = ReportGenerator(reports_dir=temp_dir)

        # Write initial report
        report_path = generator.write_report(
            run_id="run-roll",
            findings=sample_findings[:1],
            stats={"total_attempts": 10},
        )

        initial_content = report_path.read_text()
        assert "finding-001" in initial_content

        # Write updated report with more findings
        report_path = generator.write_report(
            run_id="run-roll", findings=sample_findings, stats={"total_attempts": 20}
        )

        updated_content = report_path.read_text()
        assert "finding-001" in updated_content
        assert "finding-002" in updated_content
        assert "Total Attempts: 20" in updated_content
