"""Mixture of Pipeline Experts (MoPE) prototype package.

This package contains lightweight, dependency-minimal components for experimenting
with a Transformer layer where the FFN is replaced by a mixture of executable
pipelines. The focus is on clean interfaces and inspectable traces rather than
state-of-the-art performance.
"""

from .gate import GateConfig, SimpleGate
from .pipeline import PipelineExpert, PipelineOutput, PIPELINE_REGISTRY
from .vectorizer import HashVectorizer
from .mope_layer import MoPELayer, MoPELayerConfig
from .model import MoPETransformer

__all__ = [
    "GateConfig",
    "SimpleGate",
    "PipelineExpert",
    "PipelineOutput",
    "PIPELINE_REGISTRY",
    "HashVectorizer",
    "MoPELayer",
    "MoPELayerConfig",
    "MoPETransformer",
]
