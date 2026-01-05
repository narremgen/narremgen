from __future__ import annotations
"""
narremgen.export
===============
Export utilities for turning Narremgen corpora into plain-text and LaTeX outputs.

Includes helpers to normalize Unicode, escape LaTeX, clean raw model bodies,
and build merged `.txt` and fully compilable `.tex` books from neutral and
variant CSVs, producing publication-ready narrative artifacts.
"""

from typing import List, Optional
from pathlib import Path
import datetime, re, json
import pandas as pd
from .utils import load_neutral_data
from .utils import rotate_backups
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

MAX_LATEX_BACKUPS = 100

def _normalize_unicode_for_latex(text: str) -> str:
    """
    Normalize problematic Unicode / Windows-1252 characters before LaTeX escaping.

    This helper converts a few common “fancy” characters (typographic quotes,
    dashes, ellipsis, non-breaking spaces, stray Windows-1252 glyphs) into a
    compact ASCII subset, and strips control characters that are unsafe in
    LaTeX text nodes.

    Parameters
    ----------
    text : str
        Input text. Non-string values are first converted with ``str(text)``.

    Returns
    -------
    str
        Normalized string, containing only standard ASCII punctuation plus
        the original printable characters, ready to be passed to
        :func:`latex_escape`.
    """
    if not isinstance(text, str):
        text = str(text)

    mapping = {
        "\u2018": "'",   # ‘
        "\u2019": "'",   # ’
        "\u201C": '"',   # “
        "\u201D": '"',   # ”
        "\u00B4": "'",   # ´
        "\u0060": "'",   # `
        "\u2013": "-",   # –
        "\u2014": "-",   # —
        "\u2026": "...", # …
        "\u00A0": " ",   # 
        "\u0091": "'",   # 
        "\u0092": "'",   # 
    }

    out_chars = []
    for ch in text:
        if ord(ch) < 32 and ch not in ("\n", "\t"):
            continue
        out_chars.append(mapping.get(ch, ch))

    return "".join(out_chars)


def latex_escape(text: str) -> str:
    """
    Escape the most common LaTeX special characters in a short text fragment.

    The function first normalizes the input with
    :func:`_normalize_unicode_for_latex`, then replaces characters that have
    a special meaning in LaTeX (``\\``, ``&``, ``%``, ``$``, ``#``, ``_``,
    ``{``, ``}``, ``~``, ``^``) by safe macro sequences. It is designed for
    titles, short paragraphs, and micro-fictions, not for arbitrary TeX code.

    Parameters
    ----------
    text : str
        Raw text to be embedded in LaTeX. Non-string values are converted
        using ``str(text)``.

    Returns
    -------
    str
        LaTeX-safe text where the usual special characters are escaped, and
        typical Unicode punctuation has been simplified.
    """
    if not isinstance(text, str):
        text = str(text)

    text = _normalize_unicode_for_latex(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def clean_generated_body(text: str) -> str:
    """
    Clean a generated text body for export.

    This helper removes common generation artifacts such as Markdown code fences,
    extraneous quotes, and stray headers so that downstream exports (TXT/LaTeX/DOCX/PDF)
    receive a clean, human-readable body.

    Parameters
    ----------
    text : str
        Raw text produced by the generator.

    Returns
    -------
    str
        Cleaned text.

    Notes
    -----
    The cleaning steps include:
    - remove surrounding Markdown code fences (for example: ```text\n<content>\n```),
    - trim obvious surrounding quotes (« », “ ”, or plain double quotes),
    - normalize leading/trailing whitespace.
    """

    if not isinstance(text, str):
        text = str(text)

    t = text.strip()

    if t.startswith("```") and t.endswith("```") and len(t) > 6:
        t = t[3:-3].strip()

    t = t.strip(' \n"“”«»')

    lines = [ln.rstrip() for ln in t.splitlines()]

    while lines and not lines[0].strip():
        lines.pop(0)

    TAG_PREFIX = re.compile(
        r'^\s*(SN|DE|K|STYLE|VARIANT|NEUTRAL)\s*[:=\-]\s*',
        flags=re.IGNORECASE,
    )
    TAG_LINE = re.compile(
        r'^\s*(SN|DE|K|STYLE|VARIANT|NEUTRAL)\s*[:=\-]?\s*$',
        flags=re.IGNORECASE,
    )
    BRACKET_TAG = re.compile(
        r'^\s*\[(SN|DE|K|STYLE|VARIANT|NEUTRAL)[^\]]*\]\s*$',
        flags=re.IGNORECASE,
    )

    if lines:
        first = lines[0]

        if TAG_LINE.match(first) or BRACKET_TAG.match(first):
            lines = lines[1:]
        else:
            new_first = TAG_PREFIX.sub("", first).lstrip()
            lines[0] = new_first

    t = "\n".join(lines).strip()
    return t


def build_merged_txt_from_csv(
    workdir: Path,
    variant_name: str,
    output_ext: str = "txt",
    verbose: bool = False,
) -> Path | None:
    """
    Build a flat ``merged_<variant>.txt`` file from the neutral and variant CSVs.

    The function loads the “neutral” data via :func:`load_neutral_data`:

    - ``Advice_FilteredRenumeroted_*.csv``   (Topic, Advice, Sentence, Num)
    - ``Mapping_FilteredRenumeroted_*.csv``  (Code_SN, Code_DE, Num)
    - ``corpus_for_variants.csv``            (Text, Num)

    If ``variant_name != "neutral"``, it also tries to merge
    ``VariantsGenerated_<variant>.csv`` (Num, Generated_Text). For each row:

    - if a non-empty Generated_Text exists, it is used as body,
    - otherwise the neutral ``Text`` column is used.

    The output is a plain text file that contains one LaTeX
    ``\\subsubsection*{<num>. <title>}`` header per text followed by its body
    (without additional escaping), ready to be included in a larger LaTeX
    project or post-processed.

    Parameters
    ----------
    workdir : pathlib.Path
        Root NarremGen workdir containing the neutral and variants CSV files.
    variant_name : str
        Name of the variant to export.
    output_ext : str, default "txt"
        Reserved for future use. The current implementation always writes a
        ``.txt`` file.
    verbose : bool, default False
        If True, emit debug logs about merges, fallbacks and output paths.

    Returns
    -------
    pathlib.Path or None
        Path to the created merged file, or ``None`` if there was no non-empty
        text to export.

    Raises
    ------
    ValueError
        If the neutral CSVs lack the expected columns (``Num``, ``Text``).
    """
    workdir = Path(workdir)
    advice_df, _, mapping_df, corpus_df = load_neutral_data(workdir)

    for df in (advice_df, mapping_df, corpus_df):
        if "Num" not in df.columns:
            raise ValueError(f"CSV without column 'Num' : columns={list(df.columns)}")
        df["Num"] = df["Num"].astype(int)

    if "Text" not in corpus_df.columns:
        raise ValueError(
            f"'corpus_for_variants.csv' must contain a column 'Text'. "
            f"Columns found : {list(corpus_df.columns)}"
        )

    base = advice_df.merge(mapping_df, on="Num", how="left")
    base = base.merge(corpus_df[["Num", "Text"]], on="Num", how="left")

    if base.empty:
        if verbose:
            logger_(f"[WARN] Merge (Advice + Mapping + Corpus) empty for {variant_name}.")
        return None

    base["Text"] = base["Text"].fillna("").astype(str)

    if variant_name != "neutral":
        var_csv = (
            workdir
            / "variants"
            / variant_name
            / f"VariantsGenerated_{variant_name}.csv"
        )

        if var_csv.exists():
            var_df = pd.read_csv(var_csv, sep=";")

            if "Num" not in var_df.columns or "Generated_Text" not in var_df.columns:
                raise ValueError(
                    f"{var_csv} must contain 'Num' and 'Generated_Text'. "
                    f"Columns={list(var_df.columns)}"
                )

            var_df["Num"] = var_df["Num"].astype(int)
            var_df["Generated_Text"] = var_df["Generated_Text"].fillna("").astype(str)

            base = base.merge(var_df[["Num", "Generated_Text"]], on="Num", how="left")

            gen_col = base["Generated_Text"].fillna("").astype(str)
            base["Body_Text"] = gen_col.where(gen_col.str.strip() != "", base["Text"])
        else:
            if verbose:
                logger_(f"[WARN] CSV variants not found for {variant_name}, fallback to Text neutral.")
            base["Body_Text"] = base["Text"]
    else:
        base["Body_Text"] = base["Text"]

    base = base.sort_values("Num")
    lines: list[str] = []
    for _, row in base.iterrows():
        body = str(row.get("Body_Text", "") or "").strip()
        if not body:
            continue

        num     = int(row["Num"])
        topic   = str(row.get("Topic", "") or "").strip()
        advice  = str(row.get("Advice", "") or "").strip()
        sn_code = str(row.get("Code_SN", "") or "").strip()
        de_code = str(row.get("Code_DE", "") or "").strip()

        title = advice or topic or f"Text {num}"
        suffix_parts = []
        if sn_code:
            suffix_parts.append(sn_code)
        if de_code:
            suffix_parts.append(de_code)
        if suffix_parts:
            title = f"{title} - " + " - ".join(suffix_parts)

        header = f"\\subsubsection*{{{num}. {title}}}"

        lines.append(header)
        lines.append("")
        lines.append(body)
        lines.append("")

    if not lines:
        if verbose:
            logger_(f"[WARN] No text not empty for merged_{variant_name}.")
        return None

    merged_content = "\n".join(lines)

    variant_dir = workdir / "variants" / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    base_slug = workdir.name.rsplit("_", 1)[0]
    out_path = variant_dir / f"merged_{base_slug}__{variant_name}.txt"
    rotate_backups(out_path, verbose=verbose)
    out_path.write_text(merged_content, encoding="utf-8")

    if verbose:
        logger_(f"[OK] merged_{variant_name} -> {out_path}")

    return out_path

def build_merged_tex_from_csv(
    workdir: Path,
    variant_name: str,
    document_title: str,
    author: str = "",
    output_tex: Optional[Path] = None,
    chapters_json: Optional[Path] = None,
    themes_json: Optional[Path] = None,
    add_orphans_chapter: bool = True,
    show_entry_numbers: bool = False,
    show_snde: bool = False,
    show_chapter_numbers: bool = False,
    verbose: bool = False,
) -> Path | None:
    """
    Build a standalone, fully compilable LaTeX book from NarremGen CSV corpora.

    This function extends the flat export by optionally structuring the document
    into chapters driven by external JSON definitions (themes, clusters, or
    explicit chapter mappings).

    Core behavior:
    - loads neutral CSVs and optional variant CSVs to compute the final body text;
    - generates a LaTeX book preamble, frontmatter, table of contents, and mainmatter;
    - writes each text as a numbered \\subsubsection with escaped titles and bodies.

    Optional chapter logic:
    - if ``chapters_json`` is provided and contains a ``"chapters"`` list,
      the document is structured into ``\\chapter{}`` blocks;
    - each chapter may define a ``title`` and a ``num_list`` of text identifiers;
    - if ``themes_json`` is provided, matching chapter titles may inject a short
      introductory paragraph (theme description) below the chapter header;
    - if ``assignments_json`` is provided and ``chapters_json`` is absent,
      chapters are inferred by grouping texts by theme assignments;
    - when ``add_orphans_chapter`` is True, any text not referenced by chapters
      is appended to a final fallback chapter ("Other").

    Parameters
    ----------
    workdir : pathlib.Path
        Root NarremGen workdir containing neutral and variant CSV files.
    variant_name : str
        Name of the variant to export.
    document_title : str
        Title of the LaTeX document.
    author : str, default ""
        Author name for the LaTeX ``\\author`` field.
    output_tex : pathlib.Path, optional
        Explicit output ``.tex`` path. If None, a default path is generated.
    chapters_json : pathlib.Path, optional
        JSON file defining explicit chapter structure and text groupings.
    themes_json : pathlib.Path, optional
        JSON file defining theme metadata and optional chapter introductions.
    add_orphans_chapter : bool, default True
        Whether to append unassigned texts into a final fallback chapter.
    verbose : bool, default False
        Emit debug information during export.

    Returns
    -------
    pathlib.Path or None
        Path to the generated LaTeX file, or None if no content was written.
    """

    workdir = Path(workdir)
    if not isinstance(document_title, str):
        if isinstance(document_title, (tuple, list)):
            document_title = " ".join(str(x) for x in document_title)
        else:
            document_title = str(document_title)
    document_title = document_title.strip() or f"Short Texts ({variant_name})"

    advice_df, _, mapping_df, corpus_df = load_neutral_data(workdir)

    for df in (advice_df, mapping_df, corpus_df):
        if "Num" not in df.columns:
            raise ValueError(f"CSV without column 'Num' : columns={list(df.columns)}")
        df["Num"] = df["Num"].astype(int)

    if "Text" not in corpus_df.columns:
        raise ValueError(
            f"'corpus_for_variants.csv' must contain a column 'Text'. "
            f"Column found : {list(corpus_df.columns)}"
        )

    base = advice_df.merge(mapping_df, on="Num", how="left")
    base = base.merge(corpus_df[["Num", "Text"]], on="Num", how="left")

    if base.empty:
        if verbose:
            logger_(f"[WARN] Merge (Advice + Mapping + Corpus) empty for {variant_name}.")
        return None

    base["Text"] = base["Text"].fillna("").astype(str)

    if variant_name != "neutral":
        var_csv = ( workdir / "variants" / variant_name / f"VariantsGenerated_{variant_name}.csv" )

        if var_csv.exists():
            var_df = pd.read_csv(var_csv, sep=";")

            if "Num" not in var_df.columns or "Generated_Text" not in var_df.columns:
                raise ValueError(
                    f"{var_csv} must contain 'Num' and 'Generated_Text'. "
                    f"Columns={list(var_df.columns)}"
                )

            var_df["Num"] = var_df["Num"].astype(int)
            var_df["Generated_Text"] = var_df["Generated_Text"].fillna("").astype(str)

            base = base.merge(var_df[["Num", "Generated_Text"]], on="Num", how="left")

            gen_col = base["Generated_Text"].fillna("").astype(str)
            base["Body_Text"] = gen_col.where(gen_col.str.strip() != "", base["Text"])
        else:
            if verbose:
                logger_(f"[WARN] CSV variants not found for {variant_name}, fallback to Text neutral.")
            base["Body_Text"] = base["Text"]
    else:
        base["Body_Text"] = base["Text"]

    base = base.sort_values("Num")
    chapters_data = None
    if chapters_json is not None:
        try:
            p = Path(chapters_json)
            if p.exists():
                chapters_data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            chapters_data = None

    if output_tex is None:
        base_slug = workdir.name
        variant_dir = workdir / "variants" / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        has_chapters = bool(chapters_data and isinstance(chapters_data.get("chapters", None), list))
        chapters_tag = "__chapters" if has_chapters else ""
        output_tex = variant_dir / f"book_{base_slug}__{variant_name}{chapters_tag}.tex"

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    rotate_backups(output_tex, verbose=verbose)
    
    lines: List[str] = []

    lines.append(r"\documentclass[11pt,openright]{book}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage[french,english]{babel}")
    lines.append(r"\usepackage{lmodern}")
    lines.append(r"\usepackage{setspace}")
    lines.append(r"\usepackage{geometry}")
    lines.append(r"\geometry{a4paper, margin=2.5cm}")
    lines.append(r"\usepackage{fancyhdr}")
    lines.append(r"\pagestyle{fancy}")
    lines.append(r"\usepackage[hidelinks]{hyperref}")
    lines.append(r"\onehalfspacing")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{tikz}")
    lines.append(r"\fancyhf{}")
    lines.append(r"\fancyhead[LE,RO]{\thepage}")
    lines.append(r"\fancyhead[LO]{\rightmark}")
    lines.append(r"\fancyhead[RE]{\leftmark}")
    lines.append(r"\setlength{\headheight}{14pt}")
    lines.append(r"\usepackage{titlesec}")
    if show_chapter_numbers:
        lines.append(r"\titleformat{\chapter}[hang]{\normalfont\huge\bfseries}{\thechapter.}{1em}{}")
    else:
        lines.append(r"\titleformat{\chapter}[hang]{\normalfont\huge\bfseries}{}{0pt}{}")
    lines.append(r"\titlespacing*{\chapter}{0pt}{-10pt}{20pt}")
    lines.append("")

    lines.append(r"\title{" + latex_escape(document_title) + "}")
    lines.append(r"\author{" + latex_escape(author) + "}")
    today = datetime.date.today().strftime("%d %B %Y")
    lines.append(r"\date{" + latex_escape(today) + "}")
    lines.append("")

    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\frontmatter")
    lines.append(r"\maketitle")
    lines.append(r"\clearpage")
    lines.append("")

    lines.append(r"\thispagestyle{plain}")
    lines.append(r"\section*{Warning}")
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{tikzpicture}")
    lines.append(
        r"\node[rounded corners, draw=black!50, fill=gray!5, very thick, "
        r"inner xsep=1.5em, inner ysep=1.2em, text width=0.8\textwidth] (box) {"
    )
    lines.append(
        r"\small "
        r"\textbf{Experimental content.} "
        r"This booklet is an AI-generated collection of short micro-fictions inspired by everyday advice "
        r"(e.g., cooking, organisation, work habits, health routines, and relationships), based on the selected topic."
        r"\\[0.75em]"
        r"\textbf{Not professional advice.} "
        r"It is provided for general information and creative inspiration only, and may contain errors or omissions. "
        r"For medical, legal, financial, or other professional matters, consult a qualified professional."
        r"\\[0.75em]"
        r"Generated with the {NarrEmGen} package calling several LLMs at different steps."
    )
    lines.append(r"};")
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{center}")
    lines.append(r"\vspace{1em}")
    lines.append(r"\clearpage")
    lines.append("")

    lines.append(r"\setcounter{tocdepth}{1}")
    lines.append(r"\tableofcontents")
    lines.append(r"\clearpage")
    lines.append("")
    lines.append(r"\mainmatter")
    lines.append("")

    def _emit_chapter(title: str) -> None:
        t = latex_escape(title)
        if show_chapter_numbers:
            lines.append(r"\chapter{" + t + r"}")
        else:
            lines.append(r"\chapter*{" + t + r"}")
            lines.append(r"\addcontentsline{toc}{chapter}{" + t + r"}")
            lines.append(r"\markboth{" + t + r"}{" + t + r"}")
        lines.append("")

    def _emit_one_row(row) -> None:
        body_raw = str(row.get("Body_Text", "") or "")
        body = clean_generated_body(body_raw).strip()
        if not body:
            return

        num     = int(row["Num"])
        topic   = str(row.get("Topic", "") or "").strip()
        advice  = str(row.get("Advice", "") or "").strip()
        sn_code = str(row.get("Code_SN", "") or "").strip()
        de_code = str(row.get("Code_DE", "") or "").strip()

        title = advice or topic or f"Text {num}"
        title = str(title).strip()
        title = re.sub(rf"^\s*Text\s*{num}\s*[-–—:]?\s*", "", title, flags=re.IGNORECASE)
        title = re.sub(r"^\s*\d+\s*[.)-]\s*", "", title)
        title = title.strip(" \t-–—.:")
        
        suffix_parts = []
        if show_snde:
            if sn_code:
                suffix_parts.append(sn_code)
            if de_code:
                suffix_parts.append(de_code)
        if suffix_parts:
            title = f"{title} - " + " - ".join(suffix_parts)
        safe_title = latex_escape(title)
        heading = f"{num}. {safe_title}" if show_entry_numbers else f"{safe_title}"

        lines.append(r"\section*{" + heading + r"}")
        lines.append(r"\addcontentsline{toc}{section}{" + heading + r"}")

        lines.append("")
        lines.append(latex_escape(body))
        lines.append("")

    base_by_num = {int(r["Num"]): r for _, r in base.iterrows()}

    theme_desc = {}
    if themes_json is not None:
        try:
            p = Path(themes_json)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for t in data.get("themes", []):
                    name = str(t.get("name", "") or "").strip()
                    desc = str(t.get("description", "") or "").strip()
                    if name:
                        theme_desc[name] = desc
        except Exception:
            theme_desc = {}

    chapters_data = None
    if chapters_json is not None:
        try:
            p = Path(chapters_json)
            if p.exists():
                chapters_data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            chapters_data = None

    if chapters_data and isinstance(chapters_data.get("chapters", None), list):
        used = set()
        chapters = chapters_data.get("chapters", [])

        for ch in chapters:
            raw_title = (
                ch.get("title")
                or ch.get("theme")
                or ch.get("name")
                or ch.get("chapter_id")
                or "Chapter"
            )
            ch_title = str(raw_title or "").strip() or "Chapter"
            _emit_chapter(ch_title)
            lines.append("")

            desc = theme_desc.get(ch_title, "")
            if desc:
                lines.append(r"\begin{quote}\small " + latex_escape(desc) + r"\end{quote}")
                lines.append("")

            for n in (ch.get("num_list", []) or []):
                try:
                    num = int(n)
                except Exception:
                    continue
                if num in used:
                    continue
                row = base_by_num.get(num)
                if row is None:
                    continue
                used.add(num)
                _emit_one_row(row)

        if add_orphans_chapter:
            remaining = [n for n in base_by_num.keys() if n not in used]
            if remaining:
                _emit_chapter("Other")
                for n in sorted(remaining):
                    _emit_one_row(base_by_num[n])

    else:
        for _, row in base.iterrows():
            _emit_one_row(row)

    lines.append(r"\clearpage")
    lines.append(r"\pdfbookmark[0]{About this booklet}{about-this-booklet}")
    lines.append(r"\chapter*{About this booklet}")
    lines.append(r"\markboth{About this booklet}{About this booklet}")
    lines.append(
        r"This booklet was generated using the \href{https://github.com/rpriam/NarrEmGen}{NarrEmGen} "
        r"package and a Large Language Model (LLM). "
        r"It follows a controlled narrative method (SN/DE/K): "
        r"SN and DE describe narrative and emotional structures, "
        r"while $K$ encodes elements such as character, time, place, and atmosphere. "
        r"Stylistic and formal variants are derived from an initially neutral text. "
    )
    lines.append(r"\\[0.75em]")
    lines.append(
        r"This is an experimental collection of micro-fictions inspired by everyday advice "
        r"(e.g., cooking, organisation, work habits, health routines, and relationships), "
        r"based on the selected topic."
    )
    lines.append(r"\\[0.75em]")
    lines.append(
        r"For citation, you may use a reference such as "
        r"\textit{R. Priam, ``Narrative and Emotional Structures for the Generation of Short Advisory Texts''}, "
        r"HAL \href{https://inria.hal.science/hal-05135171}{⟨hal-05135171⟩}, 2025."
    )
    lines.append(r"\\[1em]")
    lines.append(
        r"For questions, feedback or generate booklets, please visit the GitHub and PyPi pages."
    )

    lines.append("")
    lines.append(r"\backmatter")
    lines.append(r"\end{document}")
    lines.append("")

    if output_tex.exists():
        rotate_backups(output_tex, max_backups=MAX_LATEX_BACKUPS, verbose=verbose)
    output_tex.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        logger_(f"[INFO] LaTeX fusionned writen in : {output_tex}")
    return output_tex
