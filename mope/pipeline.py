"""Pipeline experts and atomic reasoning tools used by the MoPE layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

# Atomic tool signatures
PipelineStep = Callable[[str], str]


def planner(prompt: str) -> str:
    return f"plan: break down '{prompt}' into searchable claims"


def search(prompt: str) -> str:
    return f"search results for '{prompt}': [doc1, doc2]"


def reader(prompt: str) -> str:
    return f"read snippets for '{prompt}' and extract key facts"


def verifier(prompt: str) -> str:
    return f"verify facts for '{prompt}' using consensus checks"


def mult_search(prompt: str) -> str:
    return f"multi-search '{prompt}' across sources"


def compare(prompt: str) -> str:
    return f"compare findings for '{prompt}' and resolve conflicts"


@dataclass
class PipelineOutput:
    """Result of running a pipeline expert."""

    answer: str
    trace: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, str]:
        return {"answer": self.answer, "trace": "\n".join(self.trace)}


@dataclass
class PipelineExpert:
    """A chain of atomic steps representing a search/QA strategy."""

    name: str
    steps: Sequence[PipelineStep]
    description: str

    def run(self, prompt: str) -> PipelineOutput:
        trace: List[str] = []
        intermediate = prompt
        for step in self.steps:
            intermediate = step(intermediate)
            trace.append(intermediate)
        return PipelineOutput(answer=intermediate, trace=trace)


PIPELINE_REGISTRY: Dict[str, PipelineExpert] = {
    "planner-search-reader-verifier": PipelineExpert(
        name="planner-search-reader-verifier",
        steps=[planner, search, reader, verifier],
        description="Deliberate plan, search, read, and verify strategy for high-stakes answers.",
    ),
    "search-reader": PipelineExpert(
        name="search-reader",
        steps=[search, reader],
        description="Fast lookup pipeline for straightforward factual prompts.",
    ),
    "planner-search": PipelineExpert(
        name="planner-search",
        steps=[planner, search],
        description="Plan the query then search once; good for medium-difficulty prompts.",
    ),
    "multisearch-compare-verify": PipelineExpert(
        name="multisearch-compare-verify",
        steps=[mult_search, compare, verifier],
        description="Query multiple sources, compare, and verify consensus for robustness.",
    ),
}
