from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    with pd.option_context("display.float_format", lambda x: format(x, floatfmt)):
        return df.to_markdown()


def save_markdown_report(
    path: str | Path,
    title: str,
    sections: Dict[str, str],
) -> None:
    """Write a simple markdown report with a title and named sections."""
    p = Path(path)
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
