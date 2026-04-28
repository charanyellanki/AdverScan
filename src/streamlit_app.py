"""
Entry used by the official Hugging Face Streamlit Docker template::

    streamlit run src/streamlit_app.py --server.port=8501

The template only copies ``src/`` and ``requirements.txt`` by default; extend the
Dockerfile to ``COPY adverscan/`` and ``app.py`` — see ``Dockerfile.space`` and
``docs/HUGGING_FACE_SPACES.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app import main

main()
