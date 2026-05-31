"""
Central configuration for the medical imaging MCP server.

All tunable values live here rather than scattered as magic numbers. This
makes it easy to adjust behaviour via environment variables without touching
tool or client code.

Rate limit context (Semantic Scholar, as of 2025):
  - Unauthenticated: ~100 requests / 5 minutes
  - With API key:    1 request / second
Set SEMANTIC_SCHOLAR_API_KEY to unlock the higher limit.

CACHE_TTL_SECONDS prevents redundant API calls when an LLM issues identical
tool calls in one session (common when exploring a topic iteratively). Default
of 300 s matches a typical MCP session lifetime.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_key: str | None
    cache_ttl_seconds: int
    max_results_default: int
    request_timeout_seconds: float
    request_interval_seconds: float


def load_config() -> Config:
    return Config(
        api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
        max_results_default=int(os.getenv("MAX_RESULTS_DEFAULT", "10")),
        request_timeout_seconds=float(
            os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0")),
        request_interval_seconds=float(
            os.getenv("REQUEST_INTERVAL_SECONDS", "1.0")),
    )


# Fields requested from the Semantic Scholar paper endpoint by default.
#
# The API only returns fields you explicitly ask for — requesting everything
# wastes bandwidth and bloats the tokens returned to the LLM. This list covers
# the fields that are most useful for research triage. Individual tools can
# override by passing a custom list.
DEFAULT_PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "authors",
    "citationCount",
    "referenceCount",
    "externalIds",
    "publicationDate",
    "fieldsOfStudy",
    "openAccessPdf",
    "isOpenAccess",
]

DEFAULT_AUTHOR_FIELDS = [
    "authorId",
    "name",
    "affiliations",
    "paperCount",
    "citationCount",
    "hIndex",
]

# Pre-built query seeds for common medical imaging + ML subfields.
# search_medical_imaging() uses these to produce focused queries without
# requiring the caller to know the right Semantic Scholar keyword mix.
MEDICAL_IMAGING_TOPICS: dict[str, str] = {
    "segmentation": "medical image segmentation deep learning",
    "classification": "medical image classification neural network",
    "detection": "medical object detection radiology deep learning",
    "reconstruction": "medical image reconstruction MRI CT deep learning",
    "registration": "medical image registration deformable deep learning",
    "generation": "medical image synthesis generative adversarial network",
    "mri": "MRI magnetic resonance imaging deep learning",
    "ct": "CT computed tomography deep learning radiology",
    "xray": "chest X-ray radiograph classification deep learning",
    "ultrasound": "ultrasound image analysis segmentation deep learning",
    "pathology": "computational pathology histology whole slide image",
    "fundus": "fundus retinal image analysis deep learning",
    "dermoscopy": "dermoscopy skin lesion classification segmentation",
    "pet": "PET scan image analysis deep learning oncology",
    "endoscopy": "endoscopy polyp detection segmentation deep learning",
    "umap": "UMAP dimensionality reduction visualization medical imaging",
}
