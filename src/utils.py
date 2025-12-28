"""
narremgen.utils
==============
Shared utility layer for filesystem, CSV handling, and structural checks.

Centralizes helpers for workdir management, filename conventions, CSV repair
and validation, backup rotation, neutral-corpus construction, and safe LLM
generation wrappers used across all stages of the Narremgen pipeline.
"""

from .chapters import build_corpus_for_variants
import unicodedata, hashlib
from pathlib import Path
from io import StringIO
import os, re, time, glob
import pandas as pd
import shutil
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

def slugify_topic(text: str, max_len: int = 30) -> str:
    """
    Convert an arbitrary topic string into a normalized ASCII slug.

    The function:
    - removes accents and non-ASCII characters,
    - lowercases the text,
    - replaces invalid characters with spaces,
    - collapses whitespace and converts spaces to hyphens,
    - optionally truncates long slugs with a stable hash suffix.

    Parameters
    ----------
    text : str
        Raw topic string.
    max_len : int
        Maximum allowed slug length. If exceeded, a short SHA-1 hash is
        appended to preserve uniqueness.

    Returns
    -------
    str
        A filesystem-safe slug usable for directory names, filenames,
        and variant identifiers.
    """
    t = unicodedata.normalize("NFKD", text)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    slug = t.replace(" ", "-")
    if len(slug) > max_len:
        h = hashlib.sha1(slug.encode()).hexdigest()[:6]
        slug = slug[: max_len - 7].rstrip("-") + "-" + h
    return slug


def find_unique_file(directory: str, pattern: str):
    """
    Find a single matching file in a directory, falling back to the most recent one.

    The function scans `directory` with the given glob `pattern` and enforces a
    simple policy:

    - if no file matches, raise FileNotFoundError;
    - if exactly one file matches, return its path;
    - if several files match, return the most recently modified one.

    Parameters
    ----------
    directory : str
        Directory where files are searched.
    pattern : str
        Glob pattern (e.g. 'Advice_Filtered*.csv') applied inside `directory`.

    Returns
    -------
    str
        Absolute or relative path of the selected file.

    Raises
    ------
    FileNotFoundError
        If no file matches the given pattern in the directory.
    """
    matches = glob.glob(os.path.join(directory, pattern))

    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")

    if len(matches) == 1:
        return matches[0]

    matches = sorted(matches, key=os.path.getmtime, reverse=True)
    return matches[0]

def get_workdir_for_topic(
    output_dir: str | Path,
    topic_dir: str,
    create_new: bool,
) -> Path:
    """
    Select or create a working directory for a given topic run.

    Directories follow the pattern ``<topic_dir>_<k>`` inside ``output_dir``.
    The function scans existing runs, extracts the numeric suffix, and applies
    the following policy:

    - If no directory exists: create ``<topic_dir>_1`` and return it.
    - If ``create_new = False``: return the directory with the highest index.
    - If ``create_new = True``: create ``<topic_dir>_<max+1>`` and return it.

    Parameters
    ----------
    output_dir : str or Path
        Root folder where topic runs are stored.
    topic_dir : str
        Base name of the run family (without index).
    create_new : bool
        Whether to reuse the last run (False) or always create a new one (True).

    Returns
    -------
    pathlib.Path
        Path to the selected or newly created working directory.
    """
    output_dir = Path(output_dir)

    candidates: list[tuple[int, Path]] = []
    for d in output_dir.glob(f"{topic_dir}_*"):
        if not d.is_dir():
            continue
        m = re.search(r"_(\d+)$", d.name)
        if not m:
            continue
        idx = int(m.group(1))
        candidates.append((idx, d))

    if not candidates:
        workdir = output_dir / f"{topic_dir}_1"
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    candidates.sort(key=lambda t: t[0])
    last_idx, last_dir = candidates[-1]

    if not create_new:
        return last_dir

    next_idx = last_idx + 1
    workdir = output_dir / f"{topic_dir}_{next_idx}"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def build_csv_name(kind: str, stage: str, topic: str) -> str:
    """
    Construct a standardized CSV filename following the Narremgen naming convention.

    This function creates coherent filenames for all CSV files used within the
    pipeline. The convention improves readability and ensures cross-module consistency.
    Typical outputs include:
    - `Advice_Merged_<topic>.csv`
    - `Mapping_Filtered_<topic>.csv`
    - `Context_FilteredRenumeroted_<topic>.csv`

    Parameters
    ----------
    kind : str
        The file category or dataset type, such as 'advice', 'mapping', or 'context'.
    stage : str
        The pipeline stage to include in the filename (e.g., 'Merged', 'Filtered',
        'FilteredRenumeroted').
    topic : str
        The topic slug or sanitized identifier (spaces replaced by underscores).

    Returns
    -------
    str
        The fully formatted filename string, e.g. `Advice_FilteredRenumeroted_urban_walk.csv`.

    Notes
    -----
    - The returned name does not include the directory path; join it yourself when saving
    (for example: `os.path.join(output_dir, filename)`).
    - This helper guarantees consistent case and separator usage across all pipeline outputs.
    """
    slug = slugify_topic(topic)
    return f"{kind.capitalize()}_{stage}_{slug}.csv"


def save_output(text: str, filename: str, output_dir: str = "outputs", 
                ext: str = "csv", verbose=False):
    """
    Save the raw model output to disk, preferring CSV format with a text fallback.

    This helper attempts to parse the model output string as a semicolon-separated CSV.
    If parsing succeeds, the data are written as a properly formatted CSV file.
    If parsing fails (malformed content or wrong delimiter), the raw text is saved instead
    under the same base filename with the suffix `_raw.txt`.

    Parameters
    ----------
    text : str
        The text returned by the model, expected to represent a CSV table.
    filename : str
        The base filename to use when saving the output (without extension).
    output_dir : str, optional
        Directory where the file will be saved. Default is `'outputs'`.
    ext : str, optional
        Extension for structured output (default `'csv'`).
    verbose : bool, optional
        If True, prints diagnostic information about parsing errors or fallbacks.

    Returns
    -------
    tuple[str | None, int]
        `(path_to_file, num_rows)` where:
        - `path_to_file` is the saved file path (either `.csv` or `_raw.txt`),
        - `num_rows` is the number of rows successfully parsed (0 if raw text fallback is used).

    Notes
    -----
    - This function guarantees that every model call produces a persistent file,
    even when CSV parsing fails.
    - It normalizes column headers by stripping leading and trailing spaces.
    """

    os.makedirs(output_dir, exist_ok=True)
    path_csv = os.path.join(output_dir, f"{filename}.{ext}")

    if text is None:
        if verbose: logger_("!! No text generated - probably a previous API error.")
        return None, 0

    try:
        df = pd.read_csv(StringIO(text), sep=";")
        df.columns = df.columns.str.strip() 
        with open(path_csv, "w", encoding="utf-8") as f:
            df.to_csv(f, sep=";", index=False)
        return path_csv, len(df)
    except Exception as e:
        path_txt = os.path.join(output_dir, f"{filename}_raw.txt")
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write(text)
        if verbose: logger_(f"!! Error CSV, stored brut in {path_txt}")
        return path_txt, 0


def safe_generate(generate_fn, label, *args, expected_rows, max_retries=3, verbose=False, **kwargs):
    """
    Execute a generation function with retries and CSV row-count validation.

        The helper calls a generation routine, checks that the produced CSV file exists,
        and validates that the reported row count matches `expected_rows`. If the check
        fails, the call is retried up to `max_retries` times.

        Parameters
        ----------
        generate_fn : Callable
            Generation function to call. It must accept `*args` and `**kwargs` and return
            a tuple `(path, n_rows)` where `path` is the produced file path (or None) and
            `n_rows` is the number of rows written.
        label : str
            Label identifying the current stage, such as 'Advice', 'Context', or 'Mapping'.
        *args
            Positional arguments forwarded to `generate_fn`.
        expected_rows : int
            Expected number of rows in the output CSV.
        max_retries : int, default 3
            Maximum number of attempts.
        verbose : bool, default False
            If True, print progress information to stdout.
        **kwargs
            Keyword arguments forwarded to `generate_fn`.

        Returns
        -------
        tuple[str | None, int]
            `(path, n_rows)` from the final attempt.

        Raises
        ------
        RuntimeError
            If the output file is missing or the row count never matches after all retries.
        """


    for attempt in range(1, max_retries + 2):
        try:
            file_path, n_rows = generate_fn(*args, **kwargs)
        except Exception as e:
            if verbose:
                logger_(f"!! {label}: error when calling API - try number {attempt}/{max_retries}")
            time.sleep(1)
            continue 

        if not file_path or not os.path.exists(file_path):
            if verbose:
                logger_(f"!! {label}: no file generated (try number {attempt})")
            time.sleep(1)
            continue

        try:
            df = pd.read_csv(file_path, sep=";")            
        except Exception as e:
            if verbose:
                logger_(f"!!! {label}: reading impossible- try number {attempt}")
            time.sleep(1)
            continue

        if len(df) == expected_rows:
            if verbose:
                logger_(f"{label} table stored to disk ({len(df)} rows) - not yet validated")
            return file_path

        if verbose:
            logger_(f"!! {label}: {len(df)}/{expected_rows} rows - new try number ({attempt})")
        time.sleep(1)

    raise RuntimeError(f"!!! Failure in generation step {label} after {max_retries+1} tries.")


def postprocess_csv_text_basic(text: str, expected_fields: int, 
                               log_path: str = None, verbose: bool = False):
    """
    Repair malformed CSV lines in raw model output and preserve row numbering.

    This lightweight postprocessor ensures CSV consistency when the model output
    contains irregular row lengths or missing columns. Each invalid line is corrected
    by keeping the first token (usually `Num`) and padding the remaining fields with
    synthetic placeholders (`BAD_fieldX`). Optionally, all invalid lines are logged
    for inspection.

    Parameters
    ----------
    text : str
        The raw semicolon-separated text output produced by the model.
    expected_fields : int
        Number of fields expected per row.
    log_path : str | None, optional
        Path to write a text log of invalid lines. If None, no log is created.
    verbose : bool, optional
        If True, prints the number of invalid lines detected and repaired.

    Returns
    -------
    str
        The corrected CSV text string with consistent column counts across rows.

    Notes
    -----
    - This function ensures the minimal validity required to load the data with pandas.
    - Bad lines are replaced in-place with placeholder tokens so that all datasets
    remain structurally aligned for subsequent filtering and merging.
    """

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    fixed, bad_lines = [], []

    for i, line in enumerate(lines, 1):
        parts = line.split(";")
        if len(parts) != expected_fields:
            bad_lines.append((i, line))
            num_val = parts[0].strip() if parts else "MISSING"
            new_parts = [num_val] + [f"BAD_field{n}" for n in range(2, expected_fields + 1)]
            parts = new_parts
        fixed.append(";".join(parts))

    if log_path and bad_lines:
        with open(log_path, "w", encoding="utf-8") as f:
            for idx, raw in bad_lines:
                f.write(f"{idx}\t{raw}\n")
        if verbose:
            logger_(f"!! {len(bad_lines)} rows not valid logged in {log_path}")

    return "\n".join(fixed)


def merge_and_filter(output_dir: str = "outputs",
                     topic: str = "DefaultTopic",
                     verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Merge the three Merged CSV tables (Advice, Context, Mapping), detect
    corrupted rows synchronously, and produce filtered versions.

    For each dataset:
    - load the corresponding file ``<Kind>_Merged_<topic>.csv`` if it exists,
    - detect rows containing placeholder tokens such as ``BAD_``,
    - collect all ``Num`` identifiers to be excluded across datasets,
    - keep only rows whose ``Num`` is present in *all* datasets and not marked BAD,
    - write the filtered files ``<Kind>_Filtered_<topic>.csv``.

    Parameters
    ----------
    output_dir : str
        Directory containing the *_Merged_<topic>.csv files.
    topic : str
        Topic identifier used to locate the merged datasets.
    verbose : bool
        If True, prints diagnostic information about detected BAD rows
        and row counts for each dataset.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping dataset names ("advice", "context", "mapping")
        to their filtered DataFrames. Empty if no merged files were found.

    Notes
    -----
    - Filtering is *synchronized*: if a Num is invalid in any dataset, it is
      removed from all datasets to preserve alignment.
    """

    patterns = {
        "advice": "Advice_Merged",
        "context": "Context_Merged",
        "mapping": "Mapping_Merged",
    }

    merged: dict[str, pd.DataFrame] = {}

    for key, _prefix in patterns.items():
        expected_name = build_csv_name(key, "Merged", topic)
        path = os.path.join(output_dir, expected_name)

        if not os.path.exists(path):
            if verbose:
                logger_(f"!! No file found for {key}: {path}")
            continue

        df_all = pd.read_csv(path, sep=";", dtype=str)
        merged[key] = df_all
        if verbose:
            logger_(f"{key} merged ({len(df_all)} rows) from {os.path.basename(path)}")

    if not merged:
        if verbose:
            logger_("!! No merged data found for any key, aborting merge_and_filter.")
        return {}

    num_sets = [
        set(df["Num"])
        for df in merged.values()
        if "Num" in df.columns and not df.empty
    ]
    nums_all = set.intersection(*num_sets) if num_sets else set()

    bad_pat = re.compile(r"\bBAD_", re.I)
    bad_nums: set = set()

    for key, df in merged.items():
        if "Num" not in df.columns:
            continue
        mask_bad = df.apply(
            lambda s: s.astype(str).str.contains(bad_pat, na=False)
        ).any(axis=1)
        bad_nums |= set(df.loc[mask_bad, "Num"])
        if verbose and mask_bad.any():
            logger_(f"{key}: {mask_bad.sum()} rows BAD to remove")

    keep_nums = nums_all - bad_nums
    if verbose:
        logger_(
            f"Num kept = {len(keep_nums)} / {len(nums_all)} "
            f"(synchronized exclusion: {len(bad_nums)})"
        )

    filtered: dict[str, pd.DataFrame] = {}

    for key, df in merged.items():
        if "Num" not in df.columns:
            continue
        df_f = df[df["Num"].isin(keep_nums)].copy()
        filt_name = build_csv_name(key, "Filtered", topic)
        filt_path = os.path.join(output_dir, filt_name)
        df_f.to_csv(filt_path, sep=";", index=False)
        filtered[key] = df_f
        if verbose:
            logger_(f"{key} filtered -> {filt_path} ({len(df_f)})")

    return filtered


def renumerote_filtered(topic: str, output_dir="outputs", verbose=False):
    """
    Renumber filtered CSV datasets synchronously to ensure contiguous row identifiers.

    After filtering, the remaining Advice, Mapping, and Context datasets may differ slightly
    in length. This function aligns them by truncating to the smallest dataset, then assigns
    a consistent contiguous numbering sequence starting from 1. The resulting synchronized
    datasets are saved as `*_FilteredRenumeroted_<topic>.csv`.

    Parameters
    ----------
    topic : str
        Topic label used to identify the corresponding Filtered CSV files.
    output_dir : str, optional
        Directory containing the Filtered CSVs. Default is `'outputs'`.
    verbose : bool, optional
        If True, prints output file paths, number of rows written, and summary status.

    Returns
    -------
    None
        The function performs file I/O operations but does not return any object.

    Notes
    -----
    - The renumbering guarantees that all datasets share identical and consecutive `Num` values.
    - This step is essential for narrative generation, which depends on index alignment.
    - No content beyond the `Num` column is modified; only numbering is adjusted.
    """

    keys = ["advice", "mapping", "context"]
    dfs = {}
    for key in keys:
        path = os.path.join(output_dir, build_csv_name(key, "Filtered", topic))
        if not os.path.exists(path):
            if verbose:
                logger_(f"!! Missing file for {key} : {path}")
            continue
        dfs[key] = pd.read_csv(path, sep=";")

    if not dfs:
        if verbose: logger_("!!! No file Filtered found to be re-numbered.")
        return None

    n = min(len(df) for df in dfs.values())
    new_nums = range(1, n + 1)

    for key, df in dfs.items():
        if "Num" in df.columns:
            df = df.head(n).copy()
            df["Num"] = new_nums
            path_out = os.path.join(output_dir, build_csv_name(key, "FilteredRenumeroted", topic))
            df.to_csv(path_out, sep=";", index=False)
            if verbose:
                logger_(f"{key} re-numbered -> {path_out} ({n} lignes)")
    if verbose:
        logger_("Re-numbering synchronized ended.")


def audit_filtered(topic: str, output_dir="outputs"):
    """
    Perform a consistency audit on synchronized FilteredRenumeroted CSV datasets.

    This diagnostic function verifies the internal integrity of the final synchronized
    datasets (`Advice_FilteredRenumeroted_<topic>.csv`, `Mapping_FilteredRenumeroted_<topic>.csv`,
    `Context_FilteredRenumeroted_<topic>.csv`). It checks that:
    1. All three files exist.
    2. They contain the same number of rows.
    3. The `Num` column is continuous and free of duplicates.
    4. The `Num` sets are identical across all files.

    Parameters
    ----------
    topic : str
        The topic identifier corresponding to the files being audited.
    output_dir : str, optional
        Directory where the FilteredRenumeroted files are located. Default is `'outputs'`.

    Returns
    -------
    None
        Prints human-readable results to the console.

    Notes
    -----
    - The audit reports any discrepancies between the three CSVs and warns about missing files.
    - This check is meant for developer and analyst verification before launching narrative generation.
    """

    files = {k: os.path.join(output_dir, build_csv_name(k, "FilteredRenumeroted", topic)) 
             for k in ["advice","mapping","context"]}

    dfs = {k: pd.read_csv(p, sep=";") for k, p in files.items() if os.path.exists(p)}
    if len(dfs) < 3:
        logger_("!! Missing files : %s", [k for k in files if k not in dfs])
        return
    nrows = {k: len(df) for k, df in dfs.items()}
    logger_(f"number of rows : {nrows}")
    for k, df in dfs.items():
        nums = df["Num"].tolist()
        if sorted(nums) != list(range(1, len(nums) + 1)):
            logger_(f"!! {k}: Num not continuous or duplicates.")
        else:
            logger_(f"{k}: Num 1..{len(nums)} consistent.")
    common = set.intersection(*(set(df["Num"]) for df in dfs.values()))
    if all(len(df) == len(common) for df in dfs.values()):
        logger_("All files are perfectly synchronized.")
    else:
        logger_("!! Alignment inconsistency between files.")


def validate_mapping(mapping_file: str, SN_file: str, DE_file: str, 
                     verbose: bool = False) -> tuple[int, int]:
    """
    Validate Mapping file codes against official SN and DE reference tables.

    This function ensures that all `Code_SN` and `Code_DE` values in the Mapping CSV
    exist in the official reference files. Invalid codes are replaced with default
    fallbacks (`SN1` for SN, `DE1` for DE). The corrected mapping is saved in place,
    overwriting the original file.

    Parameters
    ----------
    mapping_file : str
        Path to the mapping CSV to validate. Must include columns `Code_SN` and `Code_DE`.
    SN_file : str
        Path to the official SN reference CSV. Must include at least a `Code` column.
    DE_file : str
        Path to the official DE reference CSV. Must include at least a `Code` column.
    verbose : bool, optional
        If True, prints the number of invalid codes found and corrections applied.

    Returns
    -------
    tuple[int, int]
        A tuple `(n_invalid_sn, n_invalid_de)` representing the number of invalid SN
        and DE codes that were replaced.

    Notes
    -----
    - This correction guarantees that all SN/DE codes downstream are valid and consistent.
    - The default replacement values (`SN1a`, `DE1`) correspond to safe, neutral structures.
    - The updated mapping file overwrites the original to simplify downstream use.
    """

    mapping = pd.read_csv(mapping_file, sep=";", encoding="utf-8")
    sn_ref = pd.read_csv(SN_file, sep=";", encoding="utf-8")["Code"].tolist()
    de_ref = pd.read_csv(DE_file, sep=";", encoding="utf-8")["Code"].tolist()

    invalid_sn = ~mapping["Code_SN"].isin(sn_ref)
    invalid_de = ~mapping["Code_DE"].isin(de_ref)

    n_invalid_sn = invalid_sn.sum()
    n_invalid_de = invalid_de.sum()

    if n_invalid_sn > 0:
        if verbose:
            logger_(f"!! {n_invalid_sn} codes SN not valid detected -> replaced by 'SN1'")
        mapping.loc[invalid_sn, "Code_SN"] = "SN1"

    if n_invalid_de > 0:
        if verbose:
            logger_(f"!! {n_invalid_de} codes DE not valid detected -> replaced by 'DE1'")
        mapping.loc[invalid_de, "Code_DE"] = "DE1"

    mapping.to_csv(mapping_file, sep=";", index=False, encoding="utf-8")

    if verbose and (n_invalid_sn == 0 and n_invalid_de == 0):
        logger_("All codes SN/DE from mapping are validated.")

    return n_invalid_sn, n_invalid_de


def quick_check_filtered(topic: str, output_dir="outputs", verbose=False) -> bool:
    """
    Perform a rapid structural consistency check on the final FilteredRenumeroted CSVs.

    This lightweight check verifies that the three synchronized datasets
    (Advice, Mapping, and Context) exist, share the same number of rows,
    and have identical contiguous `Num` sequences. It provides a quick way
    to confirm alignment before launching narrative generation.

    Parameters
    ----------
    topic : str
        Topic identifier used to locate the FilteredRenumeroted files.
    output_dir : str, optional
        Directory containing the synchronized CSV files. Default is `'outputs'`.
    verbose : bool, optional
        If True, prints row counts and alignment diagnostics.

    Returns
    -------
    bool
        True if all three files exist, contain the same number of rows, and have
        matching contiguous `Num` columns. False otherwise.

    Notes
    -----
    - This function is primarily used within the pipeline to abort early in case of misalignment.
    - It ensures that each `Num` corresponds to the same logical entry across datasets.
    """

    files = {
        k: os.path.join(output_dir, build_csv_name(k, "FilteredRenumeroted", topic))
        for k in ["advice", "mapping", "context"]
    }

    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        if verbose:
            logger_(f"!! Missing files for {', '.join(missing)} in {output_dir}")
        return False

    dfs = {k: pd.read_csv(p, sep=";") for k, p in files.items()}

    nrows = {k: len(df) for k, df in dfs.items()}
    if verbose:
        logger_(f"Number of rows : {nrows}")

    nset = set(nrows.values())
    if len(nset) != 1:
        if verbose:
            logger_(f"!! Incoherence in size : {nrows}")
        n_min = min(nset)
        if verbose:
            logger_(f"Truncation possible at {n_min} rows for homogeneity.")
        return False

    num_refs = dfs["advice"]["Num"].tolist()
    aligned = all(df["Num"].tolist() == num_refs for df in dfs.values())

    if not aligned:
        if verbose:
            logger_("!! Number not aligned or not continuous between files.")
        return False

    if verbose:
        logger_(f"All files FilteredRenumeroted have coherent ({len(num_refs)} rows).")

    return True


def load_neutral_data(
        workdir: Path, verbose: bool = False
    ) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Load the neutral corpus (Advice, Context, Mapping, and corpus_for_variants)
    and ensure that a local “neutral” variant is materialized.

    Steps performed:
    1. Locate the FilteredRenumeroted CSV files for Advice, Context, and Mapping.
    2. Load them into DataFrames.
    3. Ensure that ``corpus_for_variants.csv`` exists; if missing, build it
       automatically using advice + an optional merged text reference.
    4. Create a directory ``workdir/variants/neutral`` if it does not exist.
    5. Create ``VariantsGenerated_neutral.csv`` by copying the neutral texts.
    6. Copy the neutral merged text file (``merged_*.txt``) into the neutral
       variant directory for later export pipelines.

    Parameters
    ----------
    workdir : Path
        Root folder of the topic run, containing the filtered CSVs and
        optionally an existing merged narrative.
    verbose : bool
        If True, prints diagnostic information about missing files,
        auto-construction of the corpus, and created artifacts.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        (advice_df, context_df, mapping_df, corpus_df)

    Raises
    ------
    FileNotFoundError
        If any of the required FilteredRenumeroted CSV files cannot be found.
    RuntimeError
        If any of the CSV data cannot be loaded or corpus construction fails.

    Notes
    -----
    - This step guarantees that downstream variant generation always has a
      consistent “neutral” reference dataset.
    """

    workdir = Path(workdir)

    try:
        advice_csv  = Path(find_unique_file(str(workdir), "Advice_FilteredRenumeroted_*.csv"))
        context_csv = Path(find_unique_file(str(workdir), "Context_FilteredRenumeroted_*.csv"))
        mapping_csv = Path(find_unique_file(str(workdir), "Mapping_FilteredRenumeroted_*.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[NEUTRAL] Files filtered not found in {workdir}")

    corpus_csv = workdir / "corpus_for_variants.csv"
    if not corpus_csv.exists():
        if verbose:
            logger_(f"[NEUTRAL] corpus_for_variants.csv missing in {workdir}, automatic construction ...")

        try:
            merged_txt = Path(find_unique_file(str(workdir), "merged_*.txt"))
        except FileNotFoundError:
            merged_txt = None
            if verbose:
                logger_(
                    "[NEUTRAL] No file merged_*.txt found, "
                    "the corpus will be build just from the advice."
                )

        build_corpus_for_variants(
            str(advice_csv),
            str(merged_txt) if merged_txt else None,
            str(corpus_csv),
        )

        if not corpus_csv.exists():
            raise FileNotFoundError(
                f"[NEUTRAL] Impossible to build corpus_for_variants.csv in {workdir}"
            )

    try:
        advice_df  = pd.read_csv(advice_csv,  sep=";")
        context_df = pd.read_csv(context_csv, sep=";")
        mapping_df = pd.read_csv(mapping_csv, sep=";")
        corpus_df  = pd.read_csv(corpus_csv,  sep=";")
    except Exception as e:
        raise RuntimeError(f"[NEUTRAL] Impossible to load the CSV neutral")

    if verbose:
        logger_("[NEUTRAL] CSV neutral loaded : advice/context/mapping/corpus")

    neutral_dir = workdir / "variants" / "neutral"
    neutral_dir.mkdir(parents=True, exist_ok=True)

    neutral_csv = neutral_dir / "VariantsGenerated_neutral.csv"
    if not neutral_csv.exists():
        if "Num" in corpus_df.columns and "Text" in corpus_df.columns:
            neutral_df = corpus_df[["Num", "Text"]].copy()
            neutral_df = neutral_df.rename(columns={"Text": "Generated_Text"})
            neutral_df.to_csv(neutral_csv, sep=";", index=False)
            if verbose:
                logger_(f"[NEUTRAL] VariantsGenerated_neutral.csv created → {neutral_csv}")
        else:
            if verbose:
                logger_(
                    "[NEUTRAL] Impossible to create VariantsGenerated_neutral.csv : "
                    "columns 'Num' ou 'Text' missing in corpus_for_variants.csv."
                )


    merged_files = sorted(
        Path(workdir).glob("merged_*.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if merged_files:
        merged_any = merged_files[0]
        shutil.copy2(merged_any, neutral_dir / merged_any.name)

    return advice_df, context_df, mapping_df, corpus_df


def find_last_workdir(output_dir: str | Path, topic: str) -> Path:
    """
    Locate the most recent working directory for a given topic.

    Working directories follow the naming pattern ``<slugified_topic>_<k>``.
    This function:
    - slugifies the topic to determine the prefix,
    - scans ``output_dir`` for subdirectories matching ``prefix + number``,
    - extracts numeric suffixes,
    - returns the directory with the highest index.

    Parameters
    ----------
    output_dir : str or Path
        Parent directory containing all topic run folders.
    topic : str
        Raw topic string; slugified to determine the directory prefix.

    Returns
    -------
    pathlib.Path
        The path to the most recent run directory.

    Raises
    ------
    RuntimeError
        If no run directory matching the topic prefix is found.

    Notes
    -----
    - This helper is typically used to resume or inspect the latest completed run.
    """

    topic = slugify_topic(topic)

    base = Path(output_dir)
    prefix = f"{topic}_"
    candidates = []

    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(prefix):
            continue
        
        try:
            n = int(name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((n, p))

    if not candidates:
        raise RuntimeError(f"No workdir found in {output_dir} for topic '{topic}'")

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def rotate_backups(file_path: Path, max_backups: int = 50, 
 verbose: bool = False, keep_notrenum: bool = False
 ) -> None:
    """
    Create rotating backup files for a given file.

    If `file_path` exists, the function shifts existing backups upward
    (e.g., `file.bak1` becomes `file.bak2`, `file.bak2` becomes `file.bak3`),
    then writes a fresh `file.bak1` copy of the current file. Backups beyond
    `max_backups` are dropped by rotation.

    Parameters
    ----------
    file_path : str
        Path to the file to back up.
    max_backups : int, default 5
        Maximum number of backup generations to keep.
    verbose : bool, default False
        If True, print rotation steps.
    keep_notrenum : bool, default False
        If True, copy to `*.bak1` and keep the original file. If False, rename
        the original file to `*.bak1`.

    Returns
    -------
    None
    """

    if not file_path.exists():
        return
    for i in range(max_backups - 1, 0, -1):
        src = file_path.with_suffix(file_path.suffix + f".bak{i}")
        dst = file_path.with_suffix(file_path.suffix + f".bak{i + 1}")
        if src.exists():
            src.rename(dst)
    backup_path = file_path.with_suffix(file_path.suffix + ".bak1")
    if keep_notrenum:
        shutil.copy2(file_path, backup_path)
    else:
        file_path.rename(backup_path)
    if verbose:
        logger_(f"Backup created: {backup_path}")


def rotate_directory(path: str | Path, keep: int = 10,
                     verbose=False):
    path = Path(path)

    if not path.exists():
        return

    oldest = path.parent / f"{path.name}.{keep-1}"
    if oldest.exists():
        shutil.rmtree(oldest)

    for i in range(keep - 2, -1, -1):
        src = path.parent / f"{path.name}.{i}"
        dst = path.parent / f"{path.name}.{i+1}"
        if src.exists():
            src.rename(dst)

    path.rename(path.parent / f"{path.name}.0")

    if verbose:
        logger_(f"Backup directory created : {path}, with rotation {keep}!")
