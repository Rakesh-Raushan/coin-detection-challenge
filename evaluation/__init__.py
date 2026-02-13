"""
Evaluation framework for coin detection models.

This module provides production-grade evaluation capabilities including:
- Multiple detection and counting metrics
- Geometric consistency proxy metrics
- Shape diagnostics
- Failure severity analysis
- Structured artifact generation

Usage:
    from evaluation import Evaluator, EvalConfig
    
    config = EvalConfig()
    evaluator = Evaluator(config)
    artifact_paths = evaluator.run()
"""

from evaluation.config import EvalConfig
from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    compute_all_metrics,
    compute_map,
    compute_counting_metrics,
    compute_geometric_proxies,
    compute_shape_distribution,
    compute_failure_analysis,
    AllMetrics,
    MAPResults,
    CountingResults,
    GeometricProxyResults,
    ShapeDistribution,
    FailureAnalysis,
)
from evaluation.artifacts import (
    ArtifactBuilder,
    generate_artifacts,
    EvaluationSummary,
    EvaluationProvenance,
    FailureExamples,
)
from evaluation.metadata import (
    extract_model_metadata,
    extract_dataset_metadata,
    ModelMetadata,
    DatasetMetadata,
)

__all__ = [
    # Core
    "Evaluator",
    "EvalConfig",
    # Metrics
    "compute_all_metrics",
    "compute_map",
    "compute_counting_metrics",
    "compute_geometric_proxies",
    "compute_shape_distribution",
    "compute_failure_analysis",
    "AllMetrics",
    "MAPResults",
    "CountingResults",
    "GeometricProxyResults",
    "ShapeDistribution",
    "FailureAnalysis",
    # Artifacts
    "ArtifactBuilder",
    "generate_artifacts",
    "EvaluationSummary",
    "EvaluationProvenance",
    "FailureExamples",
    # Metadata
    "extract_model_metadata",
    "extract_dataset_metadata",
    "ModelMetadata",
    "DatasetMetadata",
]