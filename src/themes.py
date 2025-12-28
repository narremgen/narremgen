from __future__ import annotations
"""
narremgen.themes
===============
Theme discovery and assignment for advice corpora using LLM-based semantics.

Provides functions to extract a normalized catalog of global themes and to
assign each advice item to a theme in batched LLM calls, with robust JSON
parsing, on-disk raw dumps, and idempotent file-driven pipeline logic.
"""

import re, json
import pandas as pd
from pathlib import Path
from .llmcore import LLMConnect
from .utils import rotate_backups
from .utils import find_unique_file
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

def extract_global_themes(
    advice_csv: str,
    out_json: str = "themes_global.json",
    n_themes_min: int = 12,
    n_themes_max: int = 18,
    verbose: bool = False,
) -> str:
    """
    Extract a global theme list from an advice CSV using an LLM.

    The function reads an Advice CSV file, samples/aggregates content as needed,
    asks the model to propose a compact set of themes, then writes a JSON file
    containing a top-level `themes` list.

    Parameters
    ----------
    advice_csv : str
        Path to the advice CSV file.
    out_json : str, default "themes.json"
        Path to the JSON file to write.
    n_themes_min : int, default 8
        Minimum number of themes to request.
    n_themes_max : int, default 16
        Maximum number of themes to request.
    verbose : bool, default False
        If True, print debug information and save raw model outputs when available.

    Returns
    -------
    str
        Path to the written JSON file.

    Notes
    -----
    The resulting JSON has the structure:
    {
    "themes": [
        {
        "name": "Street safety",
        "description": "Practical habits that reduce risk while walking in public spaces.",
        "keywords": ["awareness", "lighting", "distance"]
        },
        {
        "name": "Personal boundaries",
        "description": "Ways to set limits and exit uncomfortable interactions.",
        "keywords": ["assertiveness", "exits", "de-escalation"]
        }
    ]
    }
    """

    if not (1 <= n_themes_min <= n_themes_max):
        raise ValueError("Invalid n_themes_min / n_themes_max range: require 1 <= n_themes_min <= n_themes_max.")
    df = pd.read_csv(advice_csv, sep=";")
    if "Advice" not in df.columns:
        raise ValueError("CSV must contain an 'Advice' column")

    advice_list = df["Advice"].dropna().astype(str).tolist()

    prompt = f"""
You are an expert semantic analyst. Here is a list of {len(advice_list)} short advice items:

{json.dumps(advice_list, ensure_ascii=False, indent=2)}

TASK:
1. Propose {n_themes_min} to {n_themes_max} THEMATIC GROUPS that cover all the advice above.
2. For each theme, provide:
   - name
   - one-sentence description
   - 3 to 6 keywords
3. DO NOT assign items.
4. Return strict JSON:

{{
  "themes": [
     {{
       "name": "...",
       "description": "...",
       "keywords": ["a","b","c"]
     }}
  ]
}}
"""

    if verbose:
        logger_("Extracting global themes...")

    messages = [
        {
            "role": "system",
            "content": "You are an expert semantic analyst for advice texts.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    content = LLMConnect.get_global().safe_chat_completion(
        model=LLMConnect.get_model("THEME_ANALYSIS"),
        messages=messages,
        max_tokens=8000,
    ) or ""
    
    out_path = Path(out_json)
    raw_path = out_path.with_suffix(".raw.txt")
    
    raw_path.write_text(content, encoding="utf-8")

    if verbose:
        logger_(f"[DEBUG] Raw LLM output dumped to: {raw_path}")
        logger_("[DEBUG] First 500 chars of raw output:")

    text = content.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first : last + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = content[:200].replace("\n", "\\n")
        raise ValueError(
            f"LLM did not output valid JSON. Raw content was dumped to {raw_path}. "
            f"Raw start: {preview!r}"
        ) from e

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        logger_(f"Global themes saved to: {out_json}")

    return out_json


def assign_items_to_themes(
    advice_csv: str,
    themes_json: str,
    out_json: str = "themes_assignment.json",
    batch_size: int = 40,
    allow_new_themes: bool = True,
    verbose: bool = False,
) -> str:
    """
    Assign advice items to themes using an LLM classifier.

    The function loads an advice CSV and a themes JSON produced by
    `extract_global_themes`, then classifies each advice item into one of the
    existing themes. Optionally, the model may propose new themes when none fit.

    Parameters
    ----------
    advice_csv : str
        Path to the advice CSV to classify.
    themes_json : str
        Path to the JSON file produced by `extract_global_themes`, with a
        top-level `themes` list.
    out_json : str, default "themes_assignment.json"
        Path to the JSON file where assignments and any new themes will be saved.
    batch_size : int, default 40
        Number of advice items sent to the LLM in each classification batch.
    allow_new_themes : bool, default True
        If True, allow the model to propose a new theme when no existing theme fits.
    verbose : bool, default False
        If True, log debug information and dump the beginning of each raw response
        to a `.batchN.raw.txt` file.

    Returns
    -------
    str
        Path to the written JSON file.

    Notes
    -----
    The final JSON has the structure:
    {
    "assignments": [
        {"id": 7, "theme": "Crowded intersections"},
        {"id": 80, "theme": "Night-time walking"}
    ],
    "new_themes": [
        {"name": "Unexpected detours", "description": "Route changes and improvisation for safety."}
    ]
    }
    """

    df = pd.read_csv(advice_csv, sep=";")
    if "Advice" not in df.columns:
        raise ValueError("CSV must contain an 'Advice' column")

    has_phrase = "Sentence" in df.columns

    themes = json.loads(Path(themes_json).read_text(encoding="utf-8"))["themes"]

    items = df.to_dict("records")
    assignments_all = []
    new_themes_all = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]

        if verbose:
            logger_(f"Classifying batch {i//batch_size + 1} with {len(batch)} items...")

        prompt = f"""
You are a thematic classifier.

You will receive:
1. A list of EXISTING THEMES:
{json.dumps(themes, ensure_ascii=False, indent=2)}

2. A batch of advice items:
{json.dumps(batch, ensure_ascii=False, indent=2)}

TASK:
For each item:
- Assign it to exactly ONE existing theme if the match is good.
{"- If no theme fits, propose ONE new theme." if allow_new_themes else "- You MUST choose from existing themes only (no new themes allowed)."}

Return strict JSON:
{{
"assignments": [
    {{"id": ., "theme": "."}}
],
"new_themes": [
    {{"name": ".", "description": "."}}
]
}}
"""
        messages = [
            {
                "role": "system",
                "content": "You are a thematic classifier for advice items.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        content = LLMConnect.get_global().safe_chat_completion(
            model=LLMConnect.get_model("THEME_ANALYSIS"),
            messages=messages,
            max_tokens=8000,
        ) or ""

        batch_index = i // batch_size + 1
        out_path = Path(out_json)
        raw_path = out_path.with_suffix(f".batch{batch_index}.raw.txt")
        raw_path.write_text(content, encoding="utf-8")

        if verbose:
            logger_(f"[DEBUG] Raw LLM output for batch {batch_index} dumped to: {raw_path}")
            logger_("[DEBUG] First 300 chars of raw output:")

        text = content.strip()

        fence_match = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            text = fence_match.group(1).strip()

        if not text.startswith("{"):
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                text = text[first : last + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            preview = content[:200].replace("\n", "\\n")
            raise ValueError(
                f"LLM did not output valid JSON for batch {batch_index}. "
                f"Raw content was dumped to {raw_path}. Raw start: {preview!r}"
            ) from e

        assignments_all.extend(data.get("assignments", []))

        if allow_new_themes and "new_themes" in data:
            for t in data["new_themes"]:
                new_themes_all.append(t)
                themes.append(t)

    output = {
        "assignments": assignments_all,
        "new_themes": new_themes_all,
    }

    Path(out_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        logger_(f"Assignments saved to: {out_json}")

    return out_json


def run_llm_theme_pipeline(
    workdir: Path,
    themes_json_name: str = "themes_global.json",
    assignments_json_name: str = "themes_assignment.json",
    n_themes_min: int = 8,
    n_themes_max: int = 16,
    batch_size: int = 40,
    allow_new_themes: bool = True,
    keep_notrenum:bool = False,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """
    Run the full LLM-based theme pipeline in an idempotent, file-driven way.

    The pipeline searches the NarremGen workdir for the advice CSV, manages
    backups for JSON outputs, and decides which steps to run based on the
    presence of existing files:

    - if both themes and assignments JSON files already exist, no LLM calls
      are made and their paths are simply returned;
    - if only the themes JSON exists, it is reused and only the assignment
      stage is executed;
    - if neither exists, both global theme extraction and item-to-theme
      assignment are run.

    Parameters
    ----------
    workdir : pathlib.Path
        Root directory of the NarremGen project. It must contain a single
        CSV matching the pattern "Advice_FilteredRenumeroted_*.csv".
    themes_json_name : str, default "themes_global.json"
        Filename used to store the global themes JSON inside the "themes"
        subdirectory of `workdir`.
    assignments_json_name : str, default "themes_assignment.json"
        Filename used to store item-to-theme assignments JSON in the same
        "themes" subdirectory.
    n_themes_min : int, default 8
        Minimum number of global themes requested when `extract_global_themes`
        is called.
    n_themes_max : int, default 16
        Maximum number of global themes requested when `extract_global_themes`
        is called.
    batch_size : int, default 40
        Batch size forwarded to `assign_items_to_themes` for classification.
    allow_new_themes : bool, default True
        Whether to allow the LLM to introduce new themes during assignment.
    keep_notrenum : bool, default False
        If True, copy to `*.bak1` and keep the original file. If False, rename
        the original file to `*.bak1`.
    verbose : bool, default False
        If True, log the decision path (reuse / recompute) and details of
        each stage.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        A tuple `(themes_json_path, assignments_json_path)` pointing to the
        global themes file and the item-to-theme assignments file in the
        "themes" subdirectory of `workdir`.

    Raises
    ------
    AssertionError
        If `n_themes_min` and `n_themes_max` define an invalid range.
    FileNotFoundError
        If no advice CSV matching "Advice_FilteredRenumeroted_*.csv" can
        be found in `workdir`.
    """
    if not (1 <= n_themes_min <= n_themes_max):
        raise ValueError("Invalid n_themes_min / n_themes_max range: require 1 <= n_themes_min <= n_themes_max.")
    workdir = Path(workdir)
    advice_csv = find_unique_file(workdir, "Advice_FilteredRenumeroted_*.csv")

    themes_dir = workdir / "themes"
    themes_dir.mkdir(parents=True, exist_ok=True)
    themes_json_path = themes_dir / themes_json_name
    assignments_json_path = themes_dir / assignments_json_name
    rotate_backups(themes_json_path, verbose=verbose,keep_notrenum=keep_notrenum)
    rotate_backups(assignments_json_path, verbose=verbose,keep_notrenum=keep_notrenum)

    if themes_json_path.exists() and assignments_json_path.exists():
        if verbose:
            logger_(
                "[INFO] LLM theme pipeline: existing themes and assignments found, "
                "skipping all LLM calls."
            )
        return themes_json_path, assignments_json_path

    if not themes_json_path.exists():
        if verbose:
            logger_(
                "[INFO] LLM theme pipeline: no global themes file found, "
                "extracting themes..."
            )
        extract_global_themes(
            advice_csv=str(advice_csv),
            out_json=str(themes_json_path),
            n_themes_min=n_themes_min,
            n_themes_max=n_themes_max,
            verbose=verbose,
        )
    else:
        if verbose:
            logger_(
                f"[INFO] LLM theme pipeline: reusing existing global themes file: "
                f"{themes_json_path}"
            )

    if not assignments_json_path.exists():
        if verbose:
            logger_(
                "[INFO] LLM theme pipeline: no assignment file found, "
                "running item-to-theme classification..."
            )
        assign_items_to_themes(
            advice_csv=str(advice_csv),
            themes_json=str(themes_json_path),
            out_json=str(assignments_json_path),
            batch_size=batch_size,
            allow_new_themes=allow_new_themes,
            verbose=verbose,
        )
    else:
        if verbose:
            logger_(
                f"[INFO] LLM theme pipeline: reusing existing assignments file: "
                f"{assignments_json_path}"
            )

    return themes_json_path, assignments_json_path


def ensure_single_chapter_json(workdir: Path,
                               corpus_df,
                               chapter_set_name: str = "all_texts_single",
                               source_name: str = "synthetic_all_texts",
                               filename: str = "all_in_one_file.json",
                               verbose: bool = False) -> Path:
    """
    Ensure a single-chapter JSON file exists in workdir/themes/.

    The chapter contains ALL rows of corpus_df under one unified chapter:
    - chapter_id: "All_Texts"
    - title: "All Texts"
    - num_list: [1..N]

    Returns
    -------
    Path
        The path to the JSON file.
    """
    themes_dir = workdir / "themes"
    themes_dir.mkdir(parents=True, exist_ok=True)

    chapters_json_path = themes_dir / filename

    chapters_data = {
        "chapters_set_name": chapter_set_name,
        "source": source_name,
        "chapters": [
            {
                "chapter_id": "All_Texts",
                "title": "All Texts",
                "num_list": list(range(1, corpus_df.shape[0] + 1)),
            }
        ],
    }

    if not chapters_json_path.exists():
        chapters_json_path.write_text(
            json.dumps(chapters_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if verbose and logger:
            logger_(f"[THEMES] Created {chapters_json_path}")
    else:
        if verbose and logger:
            logger_(f"[THEMES] Exists already: {chapters_json_path}")

    return chapters_json_path
