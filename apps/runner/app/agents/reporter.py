# ABOUTME: Reporter module for generating findings reports and managing promoted clusters
# ABOUTME: Handles markdown generation, PII redaction, and findings storage to JSONL

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .novelty import NoveltyDetector
from ..store.files import append_jsonl
from ..util.schemas import Finding

import logging

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Handles PII redaction from text content."""

    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            "phone": re.compile(
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b"),
            "credit_card": re.compile(
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{16}\b"
            ),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        }

        # Replacement strings
        self.replacements = {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]",
            "ssn": "[SSN_REDACTED]",
            "credit_card": "[CC_REDACTED]",
            "ip_address": "[IP_REDACTED]",
        }

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text to redact

        Returns:
            Text with PII replaced by redaction markers
        """
        if not text:
            return text

        redacted = text

        for pii_type, pattern in self.patterns.items():
            replacement = self.replacements[pii_type]
            redacted = pattern.sub(replacement, redacted)

        return redacted

    def redact_text(self, text: str) -> str:
        """Alias for redact() to maintain backward compatibility."""
        return self.redact(text)

    def has_pii(self, text: str) -> bool:
        """Check if text contains potential PII."""
        if not text:
            return False

        for pattern in self.patterns.values():
            if pattern.search(text):
                return True
        return False


class ReportGenerator:
    """Generates markdown reports from findings."""

    def __init__(
        self, reports_dir: Optional[Path] = None, redactor: Optional[PIIRedactor] = None
    ):
        self.reports_dir = Path(reports_dir) if reports_dir else Path("data/reports")
        self.redactor = redactor or PIIRedactor()
        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_report(self, run_id: str, findings: List, stats: Dict) -> Path:
        """
        Write a complete markdown report to file.

        Args:
            run_id: Run identifier
            findings: List of Finding objects
            stats: Statistics dictionary

        Returns:
            Path to the generated report file
        """
        # Generate report content
        content = self._generate_report_content(run_id, findings, stats)

        # Write to file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"run-{run_id}-{timestamp}.md"
        report_path = self.reports_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return report_path

    def _generate_report_content(self, run_id: str, findings: List, stats: Dict) -> str:
        """Generate the complete markdown report content."""
        lines = []

        # Header
        lines.append(f"# Red-Team Report: {run_id}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        if findings:
            lines.append(
                f"Discovered {len(findings)} potential vulnerabilities across multiple safety categories."
            )
        else:
            lines.append("No findings discovered in this run.")
        lines.append("")

        # Statistics
        lines.append("## Statistics")
        lines.append("")
        if "total_attempts" in stats:
            lines.append(f"Total Attempts: {stats['total_attempts']}")
        if "successful_evaluations" in stats:
            lines.append(f"Successful Evaluations: {stats['successful_evaluations']}")
        if "unique_clusters" in stats:
            lines.append(f"Unique Clusters: {stats['unique_clusters']}")
        if "total_cost_usd" in stats:
            lines.append(f"Total Cost: ${stats['total_cost_usd']:.2f}")
        if "duration_seconds" in stats:
            duration_min = stats["duration_seconds"] / 60
            lines.append(f"Duration: {duration_min:.1f} minutes")
        if "model" in stats:
            lines.append(f"Model: {stats['model']}")
        if "timestamp" in stats:
            lines.append(f"Generated: {stats['timestamp']}")
        elif "Generated:" not in "\n".join(lines):
            lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")

        # Top Findings
        lines.append("## Top Findings")
        lines.append("")
        if findings:
            # Sort by score and show top findings
            sorted_findings = sorted(findings, key=lambda f: f.score, reverse=True)
            for i, finding in enumerate(sorted_findings[:5], 1):
                lines.append(
                    f"{i}. **{finding.category}** - Score: {finding.score:.2f}, Severity: {finding.severity}"
                )
                # Redact PII in prompt and response
                redacted_prompt = self.redactor.redact(finding.prompt)
                lines.append(f"   - Prompt: {redacted_prompt[:100]}...")
        else:
            lines.append("No findings to report.")
        lines.append("")

        # Findings by Category
        lines.append("## Findings by Category")
        lines.append("")

        if findings:
            # Group by category
            by_category = {}
            for finding in findings:
                if finding.category not in by_category:
                    by_category[finding.category] = []
                by_category[finding.category].append(finding)

            # Generate sections for each category
            for category in sorted(by_category.keys()):
                lines.append(f"### {category}")
                lines.append("")

                category_findings = by_category[category]
                for finding in sorted(
                    category_findings, key=lambda f: f.score, reverse=True
                ):
                    lines.append(f"#### Finding {finding.id}")
                    lines.append(f"**Severity: {finding.severity}**")
                    lines.append(f"**Score: {finding.score:.2f}**")
                    lines.append(f"**Novelty: {finding.novelty_score:.2f}**")
                    lines.append("")

                    # Redact PII in prompt and response
                    redacted_prompt = self.redactor.redact(finding.prompt)
                    redacted_response = self.redactor.redact(finding.response)

                    lines.append(f"**Prompt:** {redacted_prompt}")
                    lines.append("")
                    lines.append(f"**Response:** {redacted_response[:200]}...")
                    lines.append("")

                    if finding.evidence:
                        lines.append("**Evidence:**")
                        for key, value in finding.evidence.items():
                            lines.append(f"- {key}: {value}")
                        lines.append("")
        else:
            lines.append("No findings to categorize.")

        lines.append("")

        # Metadata footer
        lines.append("---")
        lines.append(f"*Report generated for run {run_id}*")

        return "\n".join(lines)

    def generate_findings_section(self, findings: List[Finding]) -> str:
        """Generate markdown section for findings."""
        if not findings:
            return "## Findings\n\nNo significant findings discovered yet.\n\n"

        # Group findings by category
        by_category = {}
        for finding in findings:
            category = finding.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(finding)

        # Sort findings within each category by score
        for category in by_category:
            by_category[category].sort(key=lambda f: f.score, reverse=True)

        # Generate markdown
        md = "## Findings\n\n"

        for category, category_findings in by_category.items():
            md += f"### {category.title()} Category\n\n"

            for i, finding in enumerate(category_findings, 1):
                md += f"#### Finding {i}: {finding.description}\n\n"
                md += f"- **Score**: {finding.score:.3f}\n"
                md += f"- **Novelty Score**: {finding.novelty_score:.3f}\n"
                md += f"- **Cluster Size**: {len(finding.examples)} examples\n"
                md += f"- **Finding ID**: `{finding.finding_id}`\n"
                md += f"- **Created**: {finding.created_at}\n\n"

                # Add examples
                md += "**Examples:**\n\n"
                for j, example in enumerate(
                    finding.examples[:2], 1
                ):  # Show top 2 examples
                    md += f"**Example {j}:**\n"
                    md += f"- **Prompt**: {self.redactor.redact_text(example['prompt'])}\n"
                    md += f"- **Response**: {self.redactor.redact_text(example['response'])}\n"
                    md += f"- **Score**: {example['score']:.3f}\n"
                    if example["flags"]:
                        md += f"- **Flags**: {', '.join(example['flags'])}\n"
                    if example["reasoning"]:
                        md += f"- **Reasoning**: {self.redactor.redact_text(example['reasoning'])}\n"
                    md += "\n"

                md += "---\n\n"

        return md

    def generate_summary_stats(
        self, detector: NoveltyDetector, run_config: Optional[Dict] = None
    ) -> str:
        """Generate summary statistics section."""
        stats = detector.get_stats()

        md = "## Summary Statistics\n\n"
        md += f"- **Total Evaluations Processed**: {stats['processed_evaluations']}\n"
        md += f"- **Total Clusters**: {stats['total_clusters']}\n"
        md += f"- **Total Findings**: {stats['total_findings']}\n"

        if run_config:
            md += f"- **Run ID**: `{run_config.get('run_id', 'unknown')}`\n"
            md += f"- **Target Model**: {run_config.get('target_model', 'unknown')}\n"

        md += f"- **Generated**: {datetime.now(timezone.utc).isoformat()}\n\n"

        # Clusters by category
        if stats["clusters_by_category"]:
            md += "### Clusters by Category\n\n"
            for category, count in stats["clusters_by_category"].items():
                md += f"- **{category.title()}**: {count} clusters\n"
            md += "\n"

        # Findings by category
        if stats["findings_by_category"]:
            md += "### Findings by Category\n\n"
            for category, count in stats["findings_by_category"].items():
                md += f"- **{category.title()}**: {count} findings\n"
            md += "\n"

        return md

    def generate_methodology_section(self) -> str:
        """Generate methodology section."""
        md = "## Methodology\n\n"
        md += (
            "This report was generated using an automated red-teaming system that:\n\n"
        )
        md += "1. **Tests Generation**: Creates diverse test prompts using template mutation\n"
        md += "2. **Evaluation**: Scores responses using heuristics and LLM judges\n"
        md += "3. **Novelty Detection**: Clusters similar findings using MinHash and Jaccard similarity\n"
        md += "4. **Promotion**: Promotes high-scoring clusters to findings\n"
        md += "5. **Reporting**: Generates this markdown report with PII redaction\n\n"
        md += "### Scoring Criteria\n\n"
        md += "- **Heuristic Score**: Pattern-based detection of problematic content\n"
        md += "- **Judge Score**: LLM-based evaluation of harmfulness\n"
        md += "- **Final Score**: Weighted combination of heuristic and judge scores\n"
        md += "- **Novelty Score**: Measures uniqueness and importance of finding\n\n"
        return md


class Reporter:
    """Main reporter class for managing findings and reports."""

    def __init__(
        self,
        findings_file: str = "data/findings.jsonl",
        reports_dir: str = "data/reports",
    ):
        self.findings_file = findings_file
        self.reports_dir = reports_dir
        self.generator = ReportGenerator()

        # Ensure directories exist
        Path(self.findings_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    def promote_and_store_finding(self, finding: Finding) -> bool:
        """
        Store a promoted finding to JSONL file.

        Args:
            finding: The finding to store

        Returns:
            True if successfully stored
        """
        try:
            finding_data = finding.to_dict()
            append_jsonl(self.findings_file, finding_data)
            logger.info(f"Stored finding {finding.finding_id} to {self.findings_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to store finding {finding.finding_id}: {e}")
            return False

    def generate_report(
        self, detector: NoveltyDetector, run_id: str, run_config: Optional[Dict] = None
    ) -> str:
        """
        Generate a complete markdown report.

        Args:
            detector: The novelty detector with findings
            run_id: Run identifier
            run_config: Optional run configuration

        Returns:
            Path to generated report file
        """
        # Generate report content
        report_content = "# Red Team Findings Report\n\n"

        # Add summary stats
        report_content += self.generator.generate_summary_stats(detector, run_config)

        # Add findings
        findings = detector.get_findings()
        report_content += self.generator.generate_findings_section(findings)

        # Add methodology
        report_content += self.generator.generate_methodology_section()

        # Add appendix
        report_content += "## Appendix\n\n"
        report_content += "### Cluster Details\n\n"

        top_clusters = detector.cluster_store.get_top_clusters(limit=20)
        for i, cluster in enumerate(top_clusters, 1):
            report_content += f"**Cluster {i}** (ID: `{cluster.cluster_id}`)\n"
            report_content += f"- Category: {cluster.category}\n"
            report_content += f"- Size: {cluster.get_size()}\n"
            report_content += f"- Best Score: {cluster.best_score:.3f}\n"
            report_content += f"- Average Score: {cluster.get_average_score():.3f}\n\n"

        # Write report file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"run-{run_id}-{timestamp}.md"
        report_path = os.path.join(self.reports_dir, report_filename)

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Generated report: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Failed to write report to {report_path}: {e}")
            raise

    def get_latest_report_path(self, run_id: Optional[str] = None) -> Optional[str]:
        """Get path to the latest report file."""
        reports_dir = Path(self.reports_dir)
        if not reports_dir.exists():
            return None

        # Find report files
        pattern = f"run-{run_id}-*.md" if run_id else "run-*.md"
        report_files = list(reports_dir.glob(pattern))

        if not report_files:
            return None

        # Return most recent file
        latest_file = max(report_files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)

    def load_findings_from_file(self) -> List[Dict]:
        """Load findings from JSONL file."""
        findings = []

        if not os.path.exists(self.findings_file):
            return findings

        try:
            with open(self.findings_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        finding_data = json.loads(line)
                        findings.append(finding_data)

        except Exception as e:
            logger.error(f"Failed to load findings from {self.findings_file}: {e}")

        return findings
