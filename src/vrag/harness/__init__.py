"""Harness package.

Deliberately does *not* re-export `RAGPipeline`. `vrag.stt` depends on
`harness.retry`, and importing a submodule runs the package `__init__` first — so
re-exporting the pipeline here makes `import vrag.stt` a circular import that only
fails depending on which module the process happens to load first. Import the
submodule directly:

    from vrag.harness.pipeline import RAGPipeline
"""

from .retry import CircuitBreaker, with_retry

__all__ = ["CircuitBreaker", "with_retry"]
