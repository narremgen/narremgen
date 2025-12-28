from __future__ import annotations
"""
narremgen.chapters
===============
Tools for constructing and querying chapter structures used by the NarremGen pipeline.

Utilities for locating input files following loose filename patterns, building the canonical 
`corpus_for_variants.csv` used by variant generation, converting LLM-based theme assignments 
into a structured chapters JSON, and ensure consistent alignment between corpora, advice,themes.
"""

from typing import List, Dict
from pathlib import Path
import csv, os, re, json
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info


def build_corpus_for_variants(advice_csv: str, merged_txt: str, out_csv: str):
    """
    Build the canonical ``corpus_for_variants.csv`` used by all variant pipelines.

    The corpus aggregates, per ``Num``:

    - the base neutral text (from LaTeX/merged TXT when available, otherwise
      from the 'Advice' column),
    - the original advice sentence,
    - an optional 'Chapter' column.

    If a valid output file already exists with at least 'Num' and 'Text'
    columns, it is reused as-is.

    Parameters
    ----------
    advice_csv : str
        Path to the filtered and renumbered Advice CSV, with at least columns
        'Num' and 'Advice', separated by ';'.
    merged_txt : str
        Optional merged narratives file. If it is a CSV with 'Num' and 'Text',
        it is used directly; otherwise, the function attempts to parse LaTeX
        sections (\\subsection* or \\subsubsection*). If the file is missing
        or invalid, the advice text is used as neutral text.
    out_csv : str
        Output path for ``corpus_for_variants.csv``. Existing compatible files
        are reused instead of being overwritten.

    Returns
    -------
    str or pathlib.Path
        Path to the corpus CSV that can be consumed by later stages.

    Raises
    ------
    ValueError
        If the advice CSV is missing the 'Num' or 'Advice' columns.
    """
    out_csv = Path(out_csv)  
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv, sep=";", encoding="utf-8")
            if {"Num", "Text"}.issubset(existing.columns):
                return str(out_csv)
        except Exception:
            logger_("Failed to read existing %s; rebuilding.", out_csv, exc_info=True)

    df_adv = pd.read_csv(advice_csv, sep=";", encoding="utf-8")

    if "Num" not in df_adv.columns:
        raise ValueError(f"{advice_csv} must contain a 'Num' column.")
    if "Advice" not in df_adv.columns:
        raise ValueError(f"{advice_csv} must contain an 'Advice' column.")

    df_txt = None

    if merged_txt is not None and os.path.exists(merged_txt):
        try:
            tmp = pd.read_csv(merged_txt, sep=";", encoding="utf-8")
            if {"Num", "Text"}.issubset(tmp.columns):
                df_txt = tmp
        except Exception:
            df_txt = None

        if df_txt is None:
            with open(merged_txt, encoding="utf-8") as f:
                raw = f.read()

            pattern = r"\\sub(?:sub)?section\*?\{([^}]*)\}([\s\S]*?)(?=\\sub(?:sub)?section|\Z)"
            sections = re.findall(pattern, raw)

            rows = []
            for i, (header, body) in enumerate(sections, start=1):
                body_clean = body.strip()
                if not body_clean:
                    continue
                rows.append({
                    "Num": i,
                    "Header": header.strip(),
                    "Text": body_clean,
                })

            df_txt = pd.DataFrame(rows)
    else:
        df_txt = pd.DataFrame({
            "Num": df_adv["Num"],
            "Text": df_adv["Advice"],
        })

    df_adv["Num"] = df_adv["Num"].astype(int)
    df_txt["Num"] = df_txt["Num"].astype(int)

    df = df_txt.merge(df_adv[["Num", "Advice"]], on="Num", how="left")

    df["Advice"] = df["Advice"].fillna("")

    if "Chapter" not in df.columns:
        df["Chapter"] = ""

    df = df[["Num", "Text", "Advice", "Chapter"]]

    df.to_csv(out_csv, sep=";", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return out_csv


def create_chapters_from_theme_assignments(
    assignments_json: Path,
    out_json: Path,
    merge_small_chapters: int = 0,
    small_chapters_title: str = "Other",
) -> Path:
    """
    Build a chapters JSON file from a theme-assignments JSON.

    Reads a JSON file produced by the themes step (e.g. `themes_assignment.json`)
    and converts it into a chapters definition of the form:

    {
      "chapters_set_name": "...",
      "source": "...",
      "chapters": [
        {"chapter_id": "...", "title": "...", "num_list": [1, 2, 3]},
        ...
      ]
    }

    Parameters
    ----------
    assignments_json
        Path to the theme-assignments JSON (must contain a top-level "assignments" list
        with items holding at least "id" and "theme").
    out_json
        Output path for the generated chapters JSON.
    merge_small_chapters
        If > 0, any chapter whose `num_list` length is strictly smaller than this threshold
        is merged into a final catch-all chapter named `small_chapters_title`.
        If <= 0, no merging is performed.
    small_chapters_title
        Title of the catch-all chapter used when `merge_small_chapters > 0`.

    Returns
    -------
    Path
        The written output JSON path.
    """
    data = json.loads(assignments_json.read_text(encoding="utf-8"))
    assignments = data.get("assignments", []) or []

    by_theme: Dict[str, List[int]] = {}
    for item in assignments:
        theme_name = str(item.get("theme", "") or "").strip()
        item_id = item.get("id", None)
        if not theme_name or item_id is None:
            continue
        try:
            num_val = int(item_id)
        except Exception:
            continue
        by_theme.setdefault(theme_name, []).append(num_val)

    chapters: List[Dict] = []
    for idx, theme_name in enumerate(sorted(by_theme.keys()), start=1):
        nums_sorted = sorted(set(by_theme[theme_name]))
        safe_id = theme_name.replace(" ", "_")[:40]
        chapter_id = f"Theme_{idx}_{safe_id}"
        chapters.append(
            {
                "chapter_id": chapter_id,
                "title": theme_name,
                "num_list": nums_sorted,
            }
        )

    threshold = int(merge_small_chapters or 0)
    if threshold > 0 and chapters:
        small_nums: List[int] = []
        kept: List[Dict] = []
        for ch in chapters:
            nums = list(ch.get("num_list") or [])
            if len(nums) < threshold:
                small_nums.extend(nums)
            else:
                kept.append(ch)
        chapters = kept
        if small_nums:
            chapters.append(
                {
                    "chapter_id": "Other",
                    "title": small_chapters_title,
                    "num_list": sorted(set(small_nums)),
                }
            )

    chapters_data = {
        "chapters_set_name": "llm_themes_v1",
        "source": "llm_themes_assignments",
        "chapters": chapters,
    }

    out_json.write_text(
        json.dumps(chapters_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_json


def get_chapter_definition(chapters_data: Dict, chapter_id: str) -> Dict:
    """
    Retrieve a single chapter definition from a chapters JSON-like structure.

    The input structure must contain a top-level key "chapters" mapping to a
    list of chapter dictionaries, each with at least:

    - "chapter_id" : unique identifier,
    - "num_list"   : list of integer Num values covered by this chapter.

    Parameters
    ----------
    chapters_data : dict
        Parsed chapters JSON (as returned by ``json.load`` or similar).
    chapter_id : str
        Identifier of the chapter to retrieve.

    Returns
    -------
    dict
        The full chapter dictionary matching `chapter_id`.

    Raises
    ------
    ValueError
        If no chapter with the given `chapter_id` exists in `chapters_data`.
    """
    chapters = chapters_data.get("chapters", [])
    for chap in chapters:
        if chap.get("chapter_id") == chapter_id:
            return chap

    raise ValueError(f"Chapter '{chapter_id}' not found in the data for chapters.")
