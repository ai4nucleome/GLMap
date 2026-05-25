"""pytest configuration: put the repo root on sys.path so `import glmap.*` works
without an editable install. Tests are deliberately self-contained: no
PYTHONPATH manipulation outside this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts" / "audits"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
