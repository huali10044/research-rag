"""
Test setup for the evaluation harness.

run_eval.py imports src/rag.py, which imports chromadb and
sentence-transformers at module level. Neither is needed to test scoring,
reporting or replay, so they are stubbed when not installed. That keeps
`pytest eval/tests` runnable without downloading a model.
"""
import importlib
import sys
import types
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))


def _stub(name: str, **attrs) -> None:
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 — missing or broken install, stub either way
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub("chromadb")
_stub("chromadb.config", Settings=object)
_stub("sentence_transformers", SentenceTransformer=object)
