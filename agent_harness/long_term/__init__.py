"""Long-term memory backends (Layer 2).

`MemdirLongTermMemory` is the default (file-backed, mirrors Claude Code's
`memdir/` layout — entry file + topics + daily logs + git-root
canonicalization). `VectorLongTermMemory` is a v0.0.2 skeleton.
"""

from .memdir import MemdirLongTermMemory
from .vector import VectorLongTermMemory

__all__ = ["MemdirLongTermMemory", "VectorLongTermMemory"]
