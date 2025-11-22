from __future__ import annotations

import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def time_block(label: str) -> Iterator[float]:
    start = time.time()
    yield start
    end = time.time()
    duration = end - start
    print(f"[timing] {label}: {duration:.3f}s")
