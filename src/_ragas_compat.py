"""Import-compatibility shims for ragas on this stack (Python 3.14 + modern langchain).

Import this module before importing `ragas` anywhere.

1. ragas<0.3 unconditionally imports `langchain_community.chat_models.vertexai`
   (ragas/llms/base.py), which modern langchain-community (post-sunset restructure)
   no longer ships. We don't use Vertex AI anywhere in this project — Gemini goes
   through langchain_google_genai — so the class only needs to be importable, not
   functional.

2. ragas/executor.py calls `nest_asyncio.apply()` unconditionally at import time.
   nest_asyncio's event-loop patching is incompatible with Python 3.14's stricter
   `asyncio.timeout()` (raises "Timeout should be used inside a task"), and since
   the patch is global it breaks async HTTP calls anywhere in the process, not just
   inside ragas. nest_asyncio only exists to support re-entrant event loops (e.g.
   Jupyter, which already has one running) — we always call ragas from a plain
   synchronous entry point, so the patch is both unneeded and actively harmful here.
   No-op it before ragas can apply it.
"""

import sys
import types

import nest_asyncio

nest_asyncio.apply = lambda *args, **kwargs: None

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # noqa: N801 - matching the real class name being stubbed
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Vertex AI is not used in this project; this is an import-compat stub.")

    _shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim
