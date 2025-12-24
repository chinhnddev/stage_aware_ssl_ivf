"""
Simple logger helper.
"""

from __future__ import annotations

import sys
from pathlib import Path


class Logger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.log_file.open("a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = msg.strip()
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()
