from __future__ import annotations
"""
narremgen.variants
=================
Variant planning, rewriting, and evaluation for alternative narrative styles.

Handles extraction of reference formulas, batched rewriting of texts into
multiple styles (direct, formal, etc), optional TeX and text
exports, and neutral-vs-variant comparison statistics for each generated mode.
"""

import time
import random, re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from .llmcore import LLMConnect
from .utils import rotate_backups
from .utils import load_neutral_data
from .export import build_merged_txt_from_csv
from .export import build_merged_tex_from_csv
from .analyzestats import analyze_text_outputs
from .themes import ensure_single_chapter_json
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

MAX_RETRIES = 5

def _format_seconds(seconds: float) -> str:
    """
    Convert a positive duration in seconds into a compact human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds. Negative values are clamped to 0.

    Returns
    -------
    str
        Formatted duration such as '00m05s', '03m12s' or '01h02m00s',
        suitable for progress / ETA logs.
    """
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def batch_rewrite_from_df(
    df: pd.DataFrame,
    variant_data: dict,
    batch_size: int = 5,
    max_tokens: int = 4000,    
    retry_missing: bool = True,
    algebra: dict = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Rewrite text in batches from a DataFrame using a variant prompt.

    This function iterates over an input DataFrame, sends grouped prompts to the LLM,
    parses `[REWRITE <num>]` blocks from the response, and returns a rewritten
    DataFrame aligned with the original rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame containing the text to rewrite.
    variant_data : dict
        Variant configuration (prompt template, label, and any provider/model hints).
    batch_size : int, default 40
        Number of rows sent per LLM call.
    max_tokens : int, default 1200
        Maximum tokens for each LLM response.
    retry_missing : bool, default True
        If True, retry rewriting for rows that were not returned by the model.
    algebra : dict | None, default None
        Optional algebra data used to parameterize the variant prompt.
    verbose : bool, default False
        If True, print debug information and save raw batches when useful.

    Returns
    -------
    pandas.DataFrame
        A rewritten DataFrame with the same indexing as `df`.
    """

    variant_name = variant_data["name"]
    use_sn  = bool(variant_data.get("use_sn", False))
    use_de  = bool(variant_data.get("use_de", False))
    use_k   = bool(variant_data.get("use_k", False))
    use_ops = bool(variant_data.get("use_ops", False))
    
    prompt_file = variant_data["prompt_path"]
    base_prompt = " "
    if prompt_file:
        p = Path(prompt_file)
        try:
            base_prompt = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            cwd = Path.cwd()
            raise FileNotFoundError(
                f"Prompt file '{prompt_file}' not found in the current working directory.\n"
                f"Current working directory: {cwd}\n"
                f"- Either run the script from the directory that contains this file,\n"
                f"- or provide an explicit path (relative or absolute) to the prompt file."
            )

    if "Num" not in df.columns or "Text" not in df.columns:
        raise ValueError("The DataFrame must contain the columns 'Num' and 'Text'.")

    df = df[["Num", "Text"]].copy().dropna(subset=["Text"])
    df["Num"] = df["Num"].astype(int)
    df = df.sort_values("Num").reset_index(drop=True)

    total_texts = len(df)
    if total_texts == 0:
        return pd.DataFrame(columns=["Num", "Generated_Text"])

    all_results: list[tuple[int, str]] = []
    start_time = time.time()

    K_by_num: dict[int, dict] = {}
    struct_ops_sample: list[str] = []
    style_ops_sample: list[str] = []

    if (use_sn or use_de or use_k or use_ops) and algebra is not None:
        k_df = algebra.get("K")
        if k_df is not None and "Num" in k_df.columns:
            k_tmp = k_df.copy()
            k_tmp["Num"] = k_tmp["Num"].astype(int)
            for _, row in k_tmp.iterrows():
                K_by_num[int(row["Num"])] = row.to_dict()

        if use_ops:
            struct_df = algebra.get("STRUCT")
            style_df = algebra.get("STYLE")

            if struct_df is not None and "Op_ID" in struct_df.columns:
                sub = struct_df.sample(
                    n=min(20, len(struct_df)), random_state=random.randint(0, 10**6)
                )
                struct_ops_sample = [
                    f"- {row['Op_ID']}: {str(row.get('Description', '')).strip()}"
                    for _, row in sub.iterrows()
                ]

            if style_df is not None and "Op_ID" in style_df.columns:
                sub = style_df.sample(
                    n=min(20, len(style_df)), random_state=random.randint(0, 10**6)
                )
                style_ops_sample = [
                    f"- {row['Op_ID']}: {str(row.get('Description', '')).strip()}"
                    for _, row in sub.iterrows()
                ]

    def _run_one_batch(batch_df: pd.DataFrame, allow_missing_retry: bool) -> list[tuple[int, str]]:
        """
        Rewrite one mini-batch of texts and optionally retry missing outputs.

        The function:
        - builds the per-text metadata block (SN, DE, K) when available,
        - formats all inputs into a single prompt with numbered `[INPUT_TEXT k]`,
        - adds the operator catalogue for requiring variants,
        - calls the LLM and parses for instance the `[REWRITE k | Num=7]` sections,
        - optionally performs a single mini-batch retry for missing rewrites.

        Parameters
        ----------
        batch_df : pandas.DataFrame
            Subset of the corpus with columns 'Num' and 'Text' for this batch.
        allow_missing_retry : bool
            If True and `retry_missing` is enabled in the outer scope, a
            second call is issued only for texts that did not receive a
            valid rewrite in the first pass.

        Returns
        -------
        list[tuple[int, str]]
            List of (Num, Generated_Text) pairs for all rows in `batch_df`,
            preserving the input order and filling failures with empty strings.
        """
        local_results: list[tuple[int, str]] = []
        if batch_df.empty:
            return local_results

        nums = batch_df["Num"].astype(int).tolist()
        texts = batch_df["Text"].tolist()

        blocks = []
        for i, (num, txt) in enumerate(zip(nums, texts), start=1):
            meta_lines = []

            if K_by_num and (use_sn or use_de or use_k):
                krow = K_by_num.get(num)
                if krow:
                    code_sn = (
                        krow.get("Code_SN")
                        or krow.get("SN")
                        or krow.get("SN_code")
                        or ""
                    )
                    code_de = (
                        krow.get("Code_DE")
                        or krow.get("DE")
                        or krow.get("DE_code")
                        or ""
                    )

                    if use_sn and code_sn:
                        meta_lines.append(f"[SN] {code_sn}")
                    if use_de and code_de:
                        meta_lines.append(f"[DE] {code_de}")

                    if use_k:
                        char  = krow.get("K_char") or krow.get("K_character") or ""
                        time_ = krow.get("K_time") or ""
                        place = krow.get("K_place") or ""
                        atmos = krow.get("K_atmos") or krow.get("K_atmosphere") or ""

                        k_parts = []
                        if char:
                            k_parts.append(f"Character={char}")
                        if time_:
                            k_parts.append(f"Time={time_}")
                        if place:
                            k_parts.append(f"Place={place}")
                        if atmos:
                            k_parts.append(f"Atmosphere={atmos}")
                        if k_parts:
                            meta_lines.append("[K] " + " | ".join(k_parts))

            meta_block = ""
            if meta_lines:
                meta_block = "\n".join(meta_lines) + "\n"

            blocks.append(
                f"[INPUT_TEXT {i} | Num={num}]\n"
                f"{meta_block}"
                f"{str(txt).strip()}\n"
            )

        inputs_block = "\n".join(blocks)

        ops_block = ""
        if use_ops:
            parts = []
            if struct_ops_sample:
                parts.append("STRUCTURAL_OPERATORS:\n" + "\n".join(struct_ops_sample))
            if style_ops_sample:
                parts.append("STYLISTIC_OPERATORS:\n" + "\n".join(style_ops_sample))
            if parts:
                ops_block = (
                    "Here is the catalogue of operators you MUST choose from:\n\n"
                    + "\n\n".join(parts)
                    + "\n\n"
                )

        safety_block = """
[SAFETY_RULES]
Global safety constraints (apply to ALL rewrites):

- Never encourage or involve any behaviour with violent actions or thoughts.
- Never present clearly dangerous actions as fun, desirable, heroic, or as 
  something to try "to prove oneself".
- Whenever a situation involves a risk of serious injury (for example: high 
  falls, very high speed, deep water with strong currents, traffic, hard 
  impacts), describe it as a realistic sport or training context with 
  supervision, clear rules, and explicit safety equipment (for example: 
  jumping with a parachute, using a harness, helmet, or other protective gear). 
- Avoid any scene where a character takes an obviously life-threatening leap 
  or impact without visible, appropriate safety equipment or supervision. 
- For everyday playful movements (like jumping from one low stone to another 
  across a shallow stream), keep the description realistic and cautious, but 
  there is no need to describe full equipment unless it is naturally required 
  for the activity (for example, riding a bicycle or rollerblading). 
- When a topic sounds metaphorical (for example “jumping into the void”), 
  treat it as a figure of speech or as a fully controlled sport activity, and 
  emphasise realistic, safe behaviour in real life. 
[/SAFETY_RULES]
""".strip()

        mode_instructions = "" 

        if variant_name == "direct":
            mode_instructions = """
You will rewrite each input text into a clear, natural micro-fiction.

Goals:
- Keep the core safety advice and factual content exactly consistent.
- Preserve the overall situation and point of view.
- Improve clarity, flow, and narrative coherence.
- Use a neutral, accessible narrative voice (no strong stylistic signature).

Do NOT:
- Change the advice itself.
- Add new facts that contradict the original situation.
""".strip()

        if variant_name == "formal":
            mode_instructions = """
You will transform each input text into a concise, non-narrative advisory text
in a clear, formal tone.

Goals:
- Remove narrative and story elements (no characters, scenes, or dialogue).
- Express the same safety advice as direct recommendations or impersonal guidelines.
- Keep all factual constraints and safety content exactly consistent.
- Use clear, structured sentences suitable for a safety / guidance booklet.

Do NOT:
- Introduce characters, plot, or micro-fiction narrative.
- Change, weaken, or contradict the original advice.
- Add new speculative facts beyond what is implied in the input.
""".strip()

            core_instructions = """
You will strictly follow the system instructions above
to rewrite each input text.

- Keep the safety advice and factual content consistent.
- Respect any metadata or operators shown above each input.
- Follow the output format described below.
    """.strip()

            meta_sentence = ""
            if use_sn or use_de or use_k:
                meta_sentence = """
Metadata blocks [SN], [DE], [K] above each input describe narrative
and emotional intentions. Respect them when rewriting.
    """.strip()

            ops_sentence = ""
            if use_ops:
                ops_sentence = """
You MUST choose your transformations ONLY from the operator catalogue.
Do not invent new operators.
    """.strip()

            mode_instructions = "\n\n".join(
                p for p in [core_instructions, meta_sentence, ops_sentence] if p
            )

        extra = []

        if use_sn or use_de or use_k:
            extra.append("""
        Metadata blocks [SN], [DE], [K] above each input describe narrative
        and emotional intentions. Respect them when rewriting.
        """.strip())

        if use_ops:
            extra.append("""
        You MUST choose your transformations ONLY from the operator catalogue.
        Do not invent new operators.
        """.strip())

        if extra:
            mode_instructions = "\n\n".join(
                p for p in [mode_instructions] + extra if p.strip()
            )

        mode_instructions = mode_instructions + "\n\n" + safety_block

        format_instructions = """
Return the rewrites in the same order as the inputs, using EXACTLY this format:

[REWRITE k | Num=<same Num>]
<rewritten text k>

One block per input text, no extra commentary before or after.
""".strip()

        user_prompt = (
            f"{mode_instructions}\n\n"
            f"{ops_block}"
            "Here are the input texts:\n\n"
            f"{inputs_block}\n\n"
            f"{format_instructions}\n"
        )

        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content: Optional[str] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if verbose:
                    logger_(
                        f"[BATCH] Num from {nums[0]} to {nums[-1]} "
                        f"(size={len(nums)}), attempt {attempt}/{MAX_RETRIES}"
                    )
                content = LLMConnect.get_global().safe_chat_completion(
                    model=LLMConnect.get_model("VARIANTS_GENERATION"),
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if content is not None:
                    break
            except Exception as e:
                if verbose:
                    logger_(
                        f"[WARN] batch_rewrite_from_df: "
                        f"attempt {attempt}/{MAX_RETRIES} failed with error"
                    )

        if content is None:
            if verbose:
                logger_(
                    f"[ERROR] batch_rewrite_from_df: all {MAX_RETRIES} "
                    f"attempts failed for batch {nums[0]}..{nums[-1]}, "
                    "filling with empty strings."
                )
            for num in nums:
                local_results.append((num, ""))
            return local_results

        pattern = re.compile(
            r"(\[REWRITE[^\]]*?\])\s*(.+?)(?=(?:\[REWRITE[^\]]*?\])|\Z)",
            re.DOTALL | re.IGNORECASE,
        )

        matches = pattern.findall(content)
        if verbose:
            logger_(f"  -> {len(matches)} rewrites detected")

        rewritten_by_num: dict[int, str] = {}
        for header, txt_out in matches:
            m_num = re.search(r"Num\s*=\s*(\d+)", header, re.IGNORECASE)
            if not m_num:
                continue
            num_val = int(m_num.group(1))
            rewritten_by_num[num_val] = str(txt_out).strip()

        batch_map: dict[int, Optional[str]] = {}
        for num in nums:
            gen = rewritten_by_num.get(num)
            if isinstance(gen, str) and gen.strip():
                batch_map[num] = gen.strip()
            else:
                batch_map[num] = None

        missing_nums = [n for n, txt in batch_map.items() if txt is None]

        if missing_nums and allow_missing_retry and retry_missing:
            if verbose:
                logger_(
                    f"[WARN] Missing rewrites for {missing_nums} "
                    f"in batch {nums[0]}..{nums[-1]}, retrying mini-batch"
                )

            sub_df = (
                batch_df[batch_df["Num"].astype(int).isin(missing_nums)][["Num", "Text"]]
                .sort_values("Num")
                .reset_index(drop=True)
            )

            sub_results = _run_one_batch(sub_df, allow_missing_retry=False)
            for num_sub, gen_sub in sub_results:
                if isinstance(gen_sub, str) and gen_sub.strip():
                    batch_map[int(num_sub)] = gen_sub.strip()

        for num in nums:
            txt = batch_map.get(num) or ""
            local_results.append((num, txt))

        return local_results

    for start_idx in range(0, total_texts, batch_size):
        batch = df.iloc[start_idx : start_idx + batch_size]
        if batch.empty:
            continue

        batch_results = _run_one_batch(batch, allow_missing_retry=True)
        all_results.extend(batch_results)

        if verbose:
            processed = min(start_idx + len(batch), total_texts)
            elapsed = time.time() - start_time
            rate = elapsed / processed if processed > 0 else 0.0
            remaining = rate * (total_texts - processed) if rate > 0 else 0.0
            percent = 100.0 * processed / total_texts

            if verbose: logger_(
                                f"[PROGRESS] {processed}/{total_texts} textes "
                                f"({percent:5.1f}%) | "
                                f"elapsed={_format_seconds(elapsed)} | "
                                f"ETA≈{_format_seconds(remaining)}"
                            )

    out_df = pd.DataFrame(all_results, columns=["Num", "Generated_Text"])
    return out_df


def load_algebra_data(algebra_path: dict, sep=";") -> dict|None:
    """
    Load and validate the CSV tables used for SN/DE/K and operators.

    The `algebra_path` dictionary is expected to contain file paths under
    specific keys, which are mapped to logical tables:

    - always required: 'sn_pathdf' → 'SN', 'de_pathdf' → 'DE', 'ki_pathdf' → 'K'
    - additionally: 'opstc_pathdf' → 'STRUCT', 'opstl_pathdf' → 'STYLE'

    Parameters
    ----------
    algebra_path : dict
        Mapping from internal keys ('sn_pathdf', 'de_pathdf', 'ki_pathdf',
        'opstc_pathdf', 'opstl_pathdf') to CSV file paths.
    sep : str, default ";"
        CSV field separator used to read all tables.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary with logical keys 'SN', 'DE', 'K' and, when required,
        'STRUCT' and 'STYLE', each mapped to a DataFrame loaded from CSV.

    Raises
    ------
    ValueError
        If some required keys are missing for the given variant.
    RuntimeError
        If any of the CSV files cannot be read successfully.
    """
    if algebra_path is None:
        raise ValueError(
            f"Variant requires algebra_path, but None was provided."
        )
    key_map = {
        "sn_pathdf":    "SN",
        "de_pathdf":    "DE",
        "ki_pathdf":    "K",
        "opstc_pathdf": "STRUCT",
        "opstl_pathdf": "STYLE",
    }
    required_keys = ["sn_pathdf", "de_pathdf", "ki_pathdf","opstc_pathdf", "opstl_pathdf"]
    missing = [k for k in required_keys if algebra_path.get(k) is None]
    if missing:
        raise ValueError(
            f"Variant requires the following algebra CSV paths: {missing} "
            f"but algebra_path provided: {list(algebra_path.keys())}"
        )

    dfs = {}
    try:
        for key in required_keys:
            path = algebra_path[key]
            dfs[key_map[key]] = pd.read_csv(path, sep=sep, dtype=str)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load method' CSV for key '{key}' (path={path}) "
            f"for variant. Original error: {type(e).__name__}"
        ) from e

    return dfs


def _run_variant_pipeline(
    corpus_df: pd.DataFrame,
    variant_data: dict,
    batch_size: int,
    max_tokens: int,
    algebra_path: dict | None = None,
    verbose: bool = False,
    overwrite_existing: bool = False,
) -> Path | None:
    """
    Generate one text variant for the whole neutral corpus and write it to CSV.

    This function drives the LLM-based rewriting for a single variant type:
    it filters and orders the input corpus, loads the method' tables if
    needed, reuses any existing partial CSV, and fills in missing texts
    by calling `batch_rewrite_from_df`.

    Parameters
    ----------
    corpus_df : pandas.DataFrame
        Neutral corpus with at least columns 'Num' and 'Text'.
    variant_data : dict
        Contains the settings for this variant (name, title (for .tex), prompt_path,
        use_sn, use_de, use_k, use_ops).
    batch_size : int
        Number of texts per LLM batch.
    max_tokens : int
        Maximum number of tokens requested from the LLM per batch.
    verbose : bool, default False
        If True, log progress and warnings.
    algebra_path : dict or None, optional
        Dictionary of method' CSV paths forwarded to `load_algebra_data`
        when the variant requires SN/DE/K and operator tables.
    overwrite_existing : bool, default False
        If False and an output CSV already exists, only texts with missing or
        empty 'Generated_Text' are regenerated; if True, the CSV is rebuilt
        from scratch.

    Returns
    -------
    pathlib.Path or None
        Path to `VariantsGenerated_{variant}.csv` in `variant_path`, or None
        if the corpus is empty and nothing was generated.
    """
    variant_name = variant_data["name"].lower()
    if not {"Num", "Text"} <= set(corpus_df.columns):
        raise ValueError("corpus_df must contain 'Num' and 'Text' columns.")
    df = (
        corpus_df[["Num", "Text"]]
        .dropna(subset=["Text"])
        .assign(Num=lambda d: d["Num"].astype(int))
        .sort_values("Num")
        .reset_index(drop=True)
    )
    if df.empty:
        if verbose:
            logger_(f"[{variant_name.upper()}] corpus empty, nothing to generate.")
        return None, False

    if any(variant_data.get(flag) for flag in ("use_sn", "use_de", "use_k", "use_ops")):
        algebra = load_algebra_data(algebra_path)
    else:
        algebra = None

    variant_dir = Path(variant_data["variant_path"])
    variant_dir.mkdir(parents=True, exist_ok=True)
    out_path = variant_dir / f"VariantsGenerated_{variant_name}.csv"

    out_df = df[["Num"]].copy()
    out_df["Generated_Text"] = ""

    if out_path.exists() and not overwrite_existing:
        try:
            existing = pd.read_csv(out_path, sep=";")[["Num", "Generated_Text"]]            

            if existing["Num"].isna().any():
                raise ValueError("Some rows in existing variant CSV have non-numeric Num.")

            if existing["Num"].duplicated().any():
                dup = sorted(existing.loc[existing["Num"].duplicated(), "Num"].astype(int).unique().tolist())
                raise ValueError(
                    "Duplicate Num in existing variant CSV: "
                    + ", ".join(map(str, dup[:20]))
                    + (" ..." if len(dup) > 20 else "")
                )

            existing["Num"] = existing["Num"].astype(int)
            num_to_text = (
                existing.dropna(subset=["Generated_Text"])
                .assign(Generated_Text=lambda d: d["Generated_Text"].astype(str).str.strip())
                .set_index("Num")["Generated_Text"]
                .to_dict()
            )
            out_df["Generated_Text"] = out_df["Num"].map(num_to_text).fillna("")
        except Exception as exc:
            if verbose:
                logger_(f"[WARN] Impossible to read {out_path} : {exc}")
    elif out_path.exists() and overwrite_existing and verbose:
        logger_(f"[INFO] overwrite_existing=True, erase/rewrite of {out_path}")

    mask_missing = out_df["Generated_Text"].astype(str).str.strip().eq("")
    total = len(out_df)
    n_missing = int(mask_missing.sum())
    n_filled = int((~mask_missing).sum())

    pct_filled = 100.0 * n_filled / total if total else 0.0
    pct_missing = 100.0 * n_missing / total if total else 0.0

    filled_nums = out_df.loc[~mask_missing, "Num"].tolist()
    missing_nums = out_df.loc[mask_missing, "Num"].tolist()

    audit_path = variant_dir / "fill_audit.txt"  # workdir/variants/<variant>/
    ts = datetime.now().isoformat(timespec="seconds")
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{ts}] variant={variant_name} "
                f"total={total} filled={n_filled} ({pct_filled:5.1f}%) "
                f"missing={n_missing} ({pct_missing:5.1f}%)\n"
            )
            f.write("  filled  : " + ",".join(map(str, filled_nums)) + "\n")
            f.write("  missing : " + ",".join(map(str, missing_nums)) + "\n\n")
    except Exception as exc:
        if verbose:
            logger_(
                f"[{variant_name.upper()}] Impossible to write fill_audit.txt : {exc}"
            )

    if not mask_missing.any():
        if verbose:
            logger_(f"[{variant_name.upper()}] all the variants existed before.")
        return out_path, False

    if verbose:
        logger_(f"[{variant_name.upper()}] Texts already written : {n_filled}")
        logger_(f"[{variant_name.upper()}] Texts to be generated : {n_missing}")

    df_pending = df.loc[mask_missing, ["Num", "Text"]].reset_index(drop=True)
    new_df = batch_rewrite_from_df(
        df=df_pending,
        variant_data=variant_data,
        batch_size=batch_size,
        max_tokens=max_tokens,
        algebra=algebra,
        verbose=verbose,
    )

    if new_df is None or new_df.empty:
        if verbose:
            logger_(f"[{variant_name.upper()}] no new output generated.")
        return None, False

    new_df = new_df.assign(
        Num=lambda d: d["Num"].astype(int),
        Generated_Text=lambda d: d["Generated_Text"].fillna("").astype(str).str.strip(),
    )
    num_to_text_new = new_df.set_index("Num")["Generated_Text"].to_dict()
    out_df.loc[mask_missing, "Generated_Text"] = (
        out_df.loc[mask_missing, "Num"].map(num_to_text_new).fillna("")
    )

    if out_path.exists():
        rotate_backups(out_path, verbose=verbose)
    out_df.to_csv(out_path, sep=";", index=False, encoding="utf-8")

    if verbose:
        logger_(f"[{variant_name.upper()}] Variants written in : {out_path}")

    mask_missing_after = out_df["Generated_Text"].astype(str).str.strip().eq("")
    total_after = len(out_df)
    n_missing_after = int(mask_missing_after.sum())
    n_filled_after = int((~mask_missing_after).sum())

    pct_filled_after = 100.0 * n_filled_after / total_after if total_after else 0.0
    pct_missing_after = 100.0 * n_missing_after / total_after if total_after else 0.0

    filled_nums_after = out_df.loc[~mask_missing_after, "Num"].tolist()
    missing_nums_after = out_df.loc[mask_missing_after, "Num"].tolist()

    ts_after = datetime.now().isoformat(timespec="seconds")
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{ts_after}] AFTER  variant={variant_name} "
                f"total={total_after} filled={n_filled_after} ({pct_filled_after:5.1f}%) "
                f"missing={n_missing_after} ({pct_missing_after:5.1f}%)\n"
            )
            f.write("  filled  : " + ",".join(map(str, filled_nums_after)) + "\n")
            f.write("  missing : " + ",".join(map(str, missing_nums_after)) + "\n\n")
    except Exception as exc:
        if verbose:
            logger_(
                f"[{variant_name.upper()}] Impossible to write fill_audit.txt (after): {exc}"
            )

    return out_path, True


def run_one_variant_pipeline(
    workdir: Path,
    variant_data: dict,
    algebra_path: dict = None,
    topic:str = None,    
    batch_size: int = 20,
    max_tokens: int = 512,    
    overwrite_existing: bool =False,
    do_stats:bool =False,
    use_emo:bool =False,
    verbose: bool = False,
    ) -> tuple[Path | None, Path | None, 
           Path | None, Path | None]:
    """
    High-level end-to-end pipeline for generating and exporting one variant.

    From a NarremGen workdir, this function:
    1. Loads the neutral corpus (CSV) produced by the main narrative pipeline.
    2. Ensures a simple chapters description file `themes/all_in_one_file.json`
       exists, listing all texts in a single chapter.
    3. Calls `_run_variant_pipeline` to generate `VariantsGenerated_{vt}.csv`
       for the requested variant type.
    4. Builds merged TXT and TeX files for both 'neutral' and the variant.
    5. Optionally computes per-text and summary statistics comparing neutral
       vs variant texts.

    Parameters
    ----------
    workdir : pathlib.Path
        Root directory of the NarremGen project (contains 'neutral', 'variants', etc.).
    algebra_path : dict
        Mapping of algebra CSV paths, forwarded to `_run_variant_pipeline`
        when the variant is requiring them (see boolean options of variant_data).
    topic : str or None, optional
        High-level topic name used to build the TeX document titles.
    variant_data : dict
        Provides all variant settings, including:
        - name              (variant identifier)
        - title             (variant name for the title of the .tex)
        - prompt_path       (path to system prompt for this variant)
        - use_sn/use_de/use_k/use_ops
        - variant_path      (output directory for this variant)
    batch_size : int, default 20
        Batch size used for batched LLM calls in the variant pipeline.
    max_tokens : int, default 512
        Maximum number of tokens requested from the LLM per batch.
    overwrite_existing : bool, default False
        Whether to ignore any existing variant CSV and regenerate all texts.
    compute_textstats : bool, default True
        If True, compute neutral vs variant statistics and write them next
        to the variant CSV.
    verbose : bool, default False
        If True, enable debug logging throughout the pipeline.

    Returns
    -------
    tuple[pathlib.Path or None, pathlib.Path or None,
          pathlib.Path or None, pathlib.Path or None]
        Tuple `(variant_csv, txt_path, tex_path, summary_stats_csv)` where:
        - variant_csv : path to `VariantsGenerated_{vt}.csv` for the variant.
        - txt_path : path to the merged TXT file for the last processed style.
        - tex_path : path to the merged TeX file for the last processed style.
        - summary_stats_csv : path to the neutral vs variant stats CSV, or None
          if statistics were not computed or failed.
    """
    workdir = Path(workdir)
    variants_dir = workdir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    variant_name = variant_data["name"]
    variant_data["variant_path"] = variants_dir / f"{variant_name}"
    # rotate_directory(variant_data["variant_path"],verbose=verbose)
    use_sn  = bool(variant_data.get("use_sn", False))
    use_de  = bool(variant_data.get("use_de", False))
    use_k   = bool(variant_data.get("use_k", False))
    use_ops = bool(variant_data.get("use_ops", False))
    variant_debug = variants_dir / variant_name
    variant_debug.mkdir(parents=True, exist_ok=True)
    debug_file = variant_debug / "metadata.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write("=== VARIANT DEBUG PROMPT DUMP ===\n")
        f.write(f"Topic             : {topic}\n")
        f.write(f"Variant name      : {variant_name}\n")              
        f.write(f"Model generation  : {LLMConnect.get_model('ADVICE')}\n")
        f.write(f"Model mapping     : {LLMConnect.get_model('MAPPING')}\n")
        f.write(f"Model context     : {LLMConnect.get_model('CONTEXT')}\n")
        f.write(f"Model narrative   : {LLMConnect.get_model('NARRATIVE')}\n")
        f.write(f"Model theme       : {LLMConnect.get_model('THEME_ANALYSIS')}\n")
        f.write(f"Model variant     : {LLMConnect.get_model('VARIANTS_GENERATION')}\n")
        f.write(f"Max tokens        : {max_tokens}\n")
        f.write(f"use_sn/use_de/use_k/use_ops : "
                f"{use_sn}/{use_de}/{use_k}/{use_ops}\n")
        f.write(f"Variant path      : {variant_data['variant_path'].name}\n")  
        f.write(f"--------------------------------------------------\n")
    try:
        _, _, _, corpus_df = load_neutral_data(workdir)
    except FileNotFoundError as e:
        raise RuntimeError(
            "[VARIANTS] Impossible to load the data from neutral texts for this WORKDIR. "
            "Launch first the principal narrative pipeline to generate the CSV for neutral case."
        ) from e

    chapters_json_path = ensure_single_chapter_json(workdir=Path(workdir),
                                                    corpus_df=corpus_df,
                                                    verbose=verbose)    
    if variant_name is None: 
        if verbose:
            logger_("[INFO] Variant not given in pipeline of variants.")
        return None, None, None, None
    else:
        variant_name = variant_name.lower()
    
    if verbose:
        logger_(f"\n[INFO] MODE {variant_name.upper()} (batch rewrite) ACTIVATED ===")
    
    variant_csv, did_update = \
        _run_variant_pipeline(corpus_df=corpus_df, variant_data=variant_data, 
                              batch_size=batch_size, max_tokens=max_tokens,
                              algebra_path=algebra_path, verbose=verbose, 
                              overwrite_existing=overwrite_existing,)
    txt_path = tex_path = None
    base_slug = workdir.name.rsplit("_", 1)[0]
    neutral_txt_path = workdir / "variants" / "neutral" / f"merged_{base_slug}__neutral.txt"
    neutral_tex_path = workdir / "variants" / "neutral" / f"book_{base_slug}__neutral.tex"
    for vt in ["neutral", variant_name]:
        if vt!="neutral" or (vt=="neutral" and not neutral_txt_path.exists()):
            txt_path = build_merged_txt_from_csv(
                            workdir=workdir,
                            variant_name=vt
                        )
        if topic is not None:
            if vt!="neutral" or (vt=="neutral" and not neutral_tex_path.exists()):
                base_title  = str(topic)
                if vt == "neutral":
                    style_label = "Neutral"
                else:
                    style_label = str(variant_data.get("title", vt)).capitalize()
                document_title = f"{base_title} - Short Texts for Advice ({style_label})"
                tex_path = build_merged_tex_from_csv(
                                workdir=workdir,
                                variant_name=vt,
                                document_title = document_title,
                            )
    vt=variant_name
    summary_stats_csv = None
    summary_stats_df = None
    summary_stats_csv = variant_data["variant_path"] / f"stats_neutral_vs_{vt}.csv"
    recompute_stats = do_stats and (did_update or not summary_stats_csv.exists())
    
    if not recompute_stats:
        if verbose:
            logger_(f"[STATS] Summary already exists for {vt}, skipping recompute.")
    else:
        try:
            if corpus_df is None or corpus_df.empty:
                raise ValueError("[STATS] corpus_df is empty or None.")
            
            stats_dir = variant_data["variant_path"]
            stats_dir.mkdir(parents=True, exist_ok=True)
            neutral_per_text_csv = stats_dir / "per_text_stats_neutral.csv"
            variant_per_text_csv = stats_dir / f"per_text_stats_{vt}.csv"
            neutral_stats_df  = analyze_text_outputs(corpus_df,
                                                    text_col="Text", label="neutral", 
                                                    use_emotions=use_emo, verbose=verbose,
                                                    per_text_out=neutral_per_text_csv,)
            variant_csv = variant_data["variant_path"] / f"VariantsGenerated_{vt}.csv"
            df_variant = pd.read_csv(variant_csv,sep=";")
            if df_variant is None or df_variant.empty:
                raise ValueError(
                    f"[STATS] df_variant is empty or None for '{vt}'."
                )
            variant_stats_df = analyze_text_outputs(df_variant, text_col="Generated_Text", 
                                                    label=vt,use_emotions=use_emo,verbose=verbose,
                                                    per_text_out=variant_per_text_csv,)
            
            summary_stats_df = pd.concat([neutral_stats_df, variant_stats_df],ignore_index=True)
            summary_stats_csv = workdir / "variants" / f"{vt}" / f"stats_neutral_vs_{vt}.csv"
            summary_stats_df.to_csv(summary_stats_csv,sep=";",index=False,encoding="utf-8",)
            if verbose:
                logger_(f"[STATS] Summary neutral vs {vt} -> {summary_stats_csv}")
                #logger_(summary_stats_df)

        except Exception as e:
            if verbose:
                logger_(f"[WARN] Failure to compute the stats for {vt}")
            summary_stats_df = None

    return variant_csv, txt_path, tex_path, summary_stats_csv
