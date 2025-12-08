"""Mixture of Pipeline Experts (MoPE) prototype package.

This package contains lightweight, dependency-minimal components for experimenting
with a Transformer layer where the FFN is replaced by a mixture of executable
pipelines. The focus is on clean interfaces and inspectable traces rather than
state-of-the-art performance.
"""

from .gate import GateConfig, SimpleGate
from .model import MoPETransformer
from .mope_layer import MoPELayer, MoPELayerConfig
from .pipeline import PIPELINE_REGISTRY, PipelineExpert, PipelineOutput
from .retrieval import Document, DocumentStore, build_retrieval_pipelines, make_reader, make_retrieval_search
from .task_engine import Answer, AnswerSynthesizer, FactChecker, SearchQAFactCheckingSystem, build_task_pipelines
from .nanogpt_integration import NanoGPTMoPEAdapter, attach_mope_to_nanogpt, make_mock_nanogpt
from .vectorizer import HashVectorizer

__all__ = [
    "GateConfig",
    "SimpleGate",
    "PipelineExpert",
    "PipelineOutput",
    "PIPELINE_REGISTRY",
    "Document",
    "DocumentStore",
    "build_retrieval_pipelines",
    "make_reader",
    "make_retrieval_search",
    "Answer",
    "AnswerSynthesizer",
    "FactChecker",
    "SearchQAFactCheckingSystem",
    "build_task_pipelines",
    "NanoGPTMoPEAdapter",
    "attach_mope_to_nanogpt",
    "make_mock_nanogpt",
    "HashVectorizer",
    "MoPELayer",
    "MoPELayerConfig",
    "MoPETransformer",
]
