"""
narremgen.pipeline
=================
High-level orchestration of the multi-stage Narremgen generation workflow.

Coordinates batched CSV generation (Advice, Context, Mapping), validation and
renumbering, narrative synthesis, light analytics, and final export for a
given topic, using the shared LLMConnect configuration and utilities.
"""

import pandas as pd
import os, shutil
from .data import generate_advice, generate_mapping, generate_context
from .utils import safe_generate, validate_mapping, merge_and_filter
from .utils import renumerote_filtered, audit_filtered, build_csv_name
from .utils import quick_check_filtered
from .analyzestats import analyze_sn_de_distribution
from .narratives import generate_narratives
from .utils import load_neutral_data
from .utils import slugify_topic
from .llmcore import LLMConnect
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

def run_pipeline(topic: str,
                 workdir: str | None = None,
                 assets_dir: str | None = None,
                 csv_for_bypass_pre_gen: dict | None = None,
                 n_batches: int = 3,
                 n_per_batch: int = 20,  #>=20
                 advice_context: str | None = None,
                 output_format: str = "txt",
                 extra_instructions:str = " ",
                 dialogue_mode: str = "single",
                 verbose: bool = False,
                 )-> str|None:
    """
    Run the complete Narremgen pipeline as entry point: 
    Generates Advice, Mapping and Context tables and filtering 
    (or bypasses pre-gen with provided CSVs with checked input),
    then writes narrative batches, and basic analysis.

    This function performs the full end-to-end workflow for one topic. It:
    1. Creates a versioned subdirectory for the topic under ``output_dir``.
    2. Generates Advice, Context, and Mapping CSVs through multi-batch processing.
    3. Applies consistency checks, filtering, and renumbering.
    4. Generates narrative texts from the synchronized CSVs.
    5. Optionally exports the narratives in the requested ``output_format``.

    Parameters
    ----------
    topic : str
        Descriptive label for the generation theme (e.g., "Urban Walk").
    workdir : str, optional
        Root output directory where topic subfolders will be created.
        Default is "outputs".
    assets_dir : str or None, optional
        Directory containing the SN/DE reference CSVs and optional prompt
        assets (``style.txt``, ``examples.txt``). If None, defaults are used.
    csv_for_bypass_pre_gen: Optional dict with keys {"advice_path","mapping_path","context_path"}.
            If provided, the three CSVs must already be FilteredRenumeroted (strict Num 1..N aligned).
            They are copied into the newly created workdir under the standard expected filenames.
    n_batches : int, optional
        Number of Advice/Context/Mapping batches to generate. Default is 3.
    n_per_batch : int, optional
        Number of rows per batch. Default is 20.
    advice_context : str or None, optional
        Optional free-form background injected only into the advice generation prompt.
    output_format : str, optional
        Output format for final narratives (e.g. "docx", "txt"). Default is "docx".
    extra_instructions: str
        String character with context complement such as rules for generating advice
    dialogue_mode : str in ["none", "single", "short"], optional
        Controls the presence and structure of direct speech in neutral narratives:
        - "none": no direct dialogue at all (pure narration),
        - "single": exactly one short quoted sentence, carrying the advice,
        - "short": a short dialogue of 2–4 lines revolving around the advice.
    verbose : bool, optional
        If True, prints progress information throughout the run.

    Returns
    -------
    str
        Path to the final merged narrative output (format depends on
        ``output_format``), or an empty string if the pipeline failed early.
    """
    output_dir = "."
    try:
        if not assets_dir or not os.path.isdir(assets_dir):
            raise ValueError(
                "assets_dir must be a valid directory containing SN.csv, DE.csv, style.txt, etc."
            )
        workdir = Path(workdir) if workdir else Path(".")
        workdir.mkdir(parents=True, exist_ok=True)
        output_dir = str(workdir)

        if verbose:
            logger_(f"Run for topic '{topic}' in WORKDIR : {output_dir}")
            logger_(
                f"Models: advice={LLMConnect.get_model('ADVICE')} | "
                f"mapping={LLMConnect.get_model('MAPPING')} | "
                f"context={LLMConnect.get_model('CONTEXT')} | "
                f"narrative={LLMConnect.get_model('NARRATIVE')}"
            )
        
        def _assert_csv_header(path: str, expected_cols: list[str], label: str) -> None:
            df0 = pd.read_csv(path, sep=";", nrows=1)
            cols = [str(c).lstrip("\ufeff") for c in df0.columns.tolist()]
            if cols != expected_cols:
                raise ValueError(
                    f"{label} CSV header mismatch: expected {expected_cols} got {cols} (file={path})"
                )

        def _read_nums(path: str, label: str) -> list[int]:
            try:
                s = pd.read_csv(path, sep=";", usecols=["Num"])["Num"]
            except Exception as exc:
                raise ValueError(f"{label} CSV missing required column 'Num' (file={path})") from exc
            if s.isnull().any():
                raise ValueError(f"{label} CSV has null Num values (file={path})")
            nums = [int(x) for x in s.tolist()]
            return nums

        if csv_for_bypass_pre_gen:
            try:
                advice_src = Path(csv_for_bypass_pre_gen["advice_path"]).expanduser().resolve()
                mapping_src = Path(csv_for_bypass_pre_gen["mapping_path"]).expanduser().resolve()
                context_src = Path(csv_for_bypass_pre_gen["context_path"]).expanduser().resolve()
            except Exception as exc:
                raise ValueError(
                    "csv_for_bypass_pre_gen must contain keys: advice_path, mapping_path, context_path"
                ) from exc

            for pth, label in [(advice_src, "Advice"), (mapping_src, "Mapping"), (context_src, "Context")]:
                if not pth.exists():
                    raise FileNotFoundError(f"Bypass {label} CSV not found: {pth}")

            advice_dst = workdir / build_csv_name("advice", "FilteredRenumeroted", topic)
            mapping_dst = workdir / build_csv_name("mapping", "FilteredRenumeroted", topic)
            context_dst = workdir / build_csv_name("context", "FilteredRenumeroted", topic)

            shutil.copy2(advice_src, advice_dst)
            shutil.copy2(mapping_src, mapping_dst)
            shutil.copy2(context_src, context_dst)

            advice_file, mapping_file, context_file = str(advice_dst), str(mapping_dst), str(context_dst)

            _assert_csv_header(advice_file, ["Num", "Topic", "Advice", "Sentence"], "Advice")
            _assert_csv_header(mapping_file, ["Num", "Code_SN", "Code_DE"], "Mapping")
            _assert_csv_header(
                context_file,
                ["Num", "Character", "Presence", "Location", "Sensation", "Time", "Moment", "First_Name"],
                "Context",
            )

            nums_a = _read_nums(advice_file, "Advice")
            nums_m = _read_nums(mapping_file, "Mapping")
            nums_c = _read_nums(context_file, "Context")

            if len(nums_a) == 0 or len(nums_m) == 0 or len(nums_c) == 0:
                raise ValueError("Bypass CSV empty: no rows found (header-only file).")

            if not (nums_a == nums_m == nums_c):
                raise ValueError("Bypass CSV Num mismatch: Advice/Mapping/Context are not aligned.")

            expected = list(range(1, len(nums_a) + 1))
            if nums_a != expected:
                raise ValueError(
                    "Bypass CSV Num is not contiguous 1..N (not FilteredRenumeroted strict)."
                )

        else:
            advice_file, mapping_file, context_file = \
            generate_merge_dedup_tables(
                topic=topic, assets_dir=assets_dir,
                n_batches=n_batches, n_per_batch=n_per_batch,
                advice_context=advice_context,
                output_dir=output_dir, verbose=verbose
            )

        if not mapping_file or not os.path.exists(mapping_file):
            logger_("!! Mapping file missing or unreadable, aborting early.")
            return None

        try:
            n_batched = len(pd.read_csv(mapping_file, sep=";"))
        except Exception:
            logger_("!! Failed to read mapping_file, aborting pipeline.")
            return None

        if verbose:
            logger_(f"Final coherence check for {output_dir}")

        if not quick_check_filtered(topic, output_dir, verbose=verbose):
            if verbose: logger_(f"!! Issue detected in {output_dir}: inconsistent or missing CSV files.")
            if verbose: logger_("!!! Pipeline stopped to prevent incoherent generation.")
            return
        else:
            if verbose:
                logger_(f"Pipeline complete and verified for topic '{topic}'.")

        # qr_extra = (
        #     "If you detect that the Topic is a general question rather than a simple theme, "
        #     "interpret the Advice as a possible answer to this question. "
        #     "Each Advice comes from a specific point of view and role (profession, family member, "
        #     "friend, colleague, etc.), and the story should make this role clear. "
        #     "In that case, the main character in the story is someone who is looking for an answer "
        #     "to this question and gradually moves from not knowing to understanding. "
        #     "The narrative must clearly show this transition: at the beginning, the question is open; "
        #     "at the end, the Advice/Sentence appears as the answer the character arrives at."
        # )
        
        topic_slug = slugify_topic(topic)
        final_file = generate_narratives(
            advice_file=advice_file, context_file=context_file, mapping_file=mapping_file,
            sn_file=os.path.join(assets_dir, "SN.csv"), de_file=os.path.join(assets_dir, "DE.csv"),
            style_file=os.path.join(assets_dir, "style.txt"), examples_file=os.path.join(assets_dir, "examples.txt"),
            plain_text=True, start_line=0, end_line=n_batched, batch_size=5, language="en",
            output_dir=output_dir, final_output=f"merged_{topic_slug}.{output_format}",
            output_format=output_format, header_style="full", 
            dialogue_mode=dialogue_mode,
            extra_instructions=extra_instructions, # + "\n"+ qr_extra,
            verbose=verbose,            
        )

        try:
            load_neutral_data(Path(output_dir), verbose=verbose)
        except Exception as e:
            if verbose:
                logger_(f"[NEUTRAL] Failed to materialize files for variants/neutral")

        analyze_sn_de_distribution(mapping_file, os.path.join(assets_dir,"SN.csv"), 
                                os.path.join(assets_dir,"DE.csv"), output_dir, 
                                topic, verbose=verbose)

    except Exception as e:
        import traceback
        warning_msg = "\n!!  Something went wrong in run_pipeline."
        if verbose: logger_(warning_msg)
        # if verbose: logger_("Error type: %s", type(e).__name__)
        # if verbose: logger_("Message: %s", e)
        traceback.print_exc()
        log_path = os.path.join(output_dir, "error_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(warning_msg + "\n")
            traceback.print_exc(file=f)
        if verbose: logger_(f"(Error logged in {log_path})")
        return None
    
    return final_file


def generate_all_csv_batches(topic: str,
                             assets_dir: str,
                             n_batches: int = 5,
                             n_per_batch: int = 20,
                             advice_context: str | None = None,
                             output_dir: str = "outputs",
                             verbose: bool = False):
    """
    Generate Advice, Context, and Mapping CSVs across multiple batches
    with built-in validation.

    For each batch, the function:
    1. Generates Advice entries for the given topic.
    2. Generates the corresponding Context aligned by ``Num``.
    3. Generates the Mapping table linking each ``Num`` to SN/DE codes.
    4. Validates the Mapping against the official SN/DE tables.
    5. Merges and deduplicates the per-batch CSVs when all batches are done.

    Parameters
    ----------
    llm : LLMConnect
        Unified LLM manager used for all model calls in this pipeline stage.
    topic : str
        Topic name or label used in prompts and filenames.
    assets_dir : str
        Directory containing the official SN and DE reference files.
    n_batches : int, optional
        Number of batches to generate. Default is 5.
    n_per_batch : int, optional
        Number of rows per batch. Default is 20.
    advice_context : str or None, optional
        Optional free-form background injected only into the advice generation prompt.
    output_dir : str, optional
        Root output directory for storing intermediate and merged CSVs.
        Default is "outputs".
    model_advice : str, optional
        Model used for Advice generation. Default is "gpt-4o-mini".
    model_mapping : str, optional
        Model used for Mapping generation (e.g. "o3"). Default is "o3".
    model_context : str, optional
        Model used for Context generation. Default is "gpt-4o-mini".
    verbose : bool, optional
        If True, prints detailed progress logs for each batch and stage.

    Returns
    -------
    dict
        A small summary dictionary with paths to the merged Advice,
        Context, and Mapping CSVs.
    """

    os.makedirs(output_dir, exist_ok=True)
    SN_file = os.path.join(assets_dir, "SN.csv")
    DE_file = os.path.join(assets_dir, "DE.csv")
    
    batch_dir = os.path.join(output_dir, "batches_csv")
    os.makedirs(batch_dir, exist_ok=True)

    advice_paths, mapping_paths, context_paths = [], [], []
    failed_batches = []

    for i in range(n_batches):
        subtopic = f"{topic}_(batch_{i+1})"
        if verbose:
            logger_(f"\n=== Generation of batch {i+1}/{n_batches} ===")

        advice_file = None
        try:
            advice_file = safe_generate(
                generate_advice, "Advice", subtopic,
                expected_rows=n_per_batch, 
                n_advice=n_per_batch,
                advice_context=advice_context,
                output_dir=batch_dir, verbose=verbose
            )
        except RuntimeError:
            if verbose: logger_(f"!!! Batch {i+1} abandoned (Advice failed after internal retries).")
            failed_batches.append((i+1, "Advice"))
            continue
        if not advice_file:
            if verbose:
                logger_(f"!!! Batch {i+1} abandoned (Advice KO after 3 tries).")
            failed_batches.append((i+1, "Advice"))
            continue
        advice_paths.append(advice_file)

        context_file = None
        try:
            context_file = safe_generate(
                generate_context, "Context", advice_file,
                expected_rows=n_per_batch,
                output_dir=batch_dir, verbose=verbose
            )
        except RuntimeError:
            if verbose: logger_(f"!!! Batch {i+1} abandoned (Context failed after internal retries).")
            failed_batches.append((i+1, "Context"))
            continue
        if not context_file:
            if verbose:
                logger_(f"!!! Batch {i+1} abandoned (Context KO after 3 tries).")
            failed_batches.append((i+1, "Context"))
            continue
        context_paths.append(context_file)

        mapping_file = None
        success = False
        for attempt in range(1, 4):  
            try:
                mapping_file = safe_generate(
                    generate_mapping, "Mapping", advice_file,
                    SN_file, DE_file,
                    expected_rows=n_per_batch,
                    output_dir=batch_dir,
                    verbose=verbose,
                    max_retries=3,   
                )
            except RuntimeError:
                if verbose:
                    logger_(f"!!! Mapping failed (internal retries) at try {attempt} for batch {i+1}.")
                continue

            if not mapping_file:
                if verbose:
                    logger_(f"!! Mapping failed (try number {attempt}) - re-launch Mapping...")
                continue

            n_invalid_sn, n_invalid_de = validate_mapping(
                mapping_file, SN_file, DE_file, verbose=verbose
            )

            if n_invalid_sn == 0 and n_invalid_de == 0:
                success = True
                mapping_paths.append(mapping_file)
                if verbose:
                    logger_(f"Mapping valid (batch {i+1}) at try number {attempt}.")
                break
            else:
                if verbose:
                    logger_(
                        f"!! Mapping not valid (batch {i+1}) : "
                        f"{n_invalid_sn} SN, {n_invalid_de} DE - re-launch Mapping..."
                    )

        if not success:
            if verbose:
                logger_(f"!!! Batch {i+1} abandoned (Mapping KO after 3 tries).")
            failed_batches.append((i + 1, "Mapping"))
            continue


    if failed_batches:
        log_path = os.path.join(batch_dir, "failed_batches.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for num, stage in failed_batches:
                f.write(f"{num}\t{stage}\n")
        if verbose: logger_(f"\n !!  {len(failed_batches)} batch(s) have failed : {failed_batches}")
        if verbose: logger_(f"Details stored in {log_path}")

    def list_batch_files(prefix):
        return sorted(
            os.path.join(batch_dir, f)
            for f in os.listdir(batch_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        )

    adv_files = list_batch_files("Advice_")
    map_files = list_batch_files("Mapping_")
    ctx_files = list_batch_files("Context_")

    if verbose:
        logger_(f"\n Batch filed found : "
            f"{len(adv_files)} advices, {len(map_files)} mappings, {len(ctx_files)} contexts")

    if not adv_files or not map_files or not ctx_files:
        raise RuntimeError(f"No batch file found in {batch_dir}")

    all_adv = pd.concat([pd.read_csv(f, sep=";") for f in adv_files], ignore_index=True)
    all_map = pd.concat([pd.read_csv(f, sep=";") for f in map_files], ignore_index=True)
    all_ctx = pd.concat([pd.read_csv(f, sep=";") for f in ctx_files], ignore_index=True)

    if len({len(all_adv), len(all_map), len(all_ctx)}) != 1:
        if verbose: logger_(f"!! Incoherence at post-merge : "
            f"adv={len(all_adv)}, map={len(all_map)}, ctx={len(all_ctx)}")
        min_len = min(len(all_adv), len(all_map), len(all_ctx))
        all_adv, all_map, all_ctx = all_adv.head(min_len), all_map.head(min_len), all_ctx.head(min_len)

    new_nums = range(1, len(all_adv) + 1)
    for df in (all_adv, all_map, all_ctx):
        if "Num" in df.columns:
            df["Num"] = new_nums

    all_adv.to_csv(os.path.join(output_dir, build_csv_name("advice", "Merged", topic)), sep=";", index=False)
    all_map.to_csv(os.path.join(output_dir, build_csv_name("mapping", "Merged", topic)), sep=";", index=False)
    all_ctx.to_csv(os.path.join(output_dir, build_csv_name("context", "Merged", topic)), sep=";", index=False)

    if verbose:
        logger_(f"Global merge done and re-numbered for topic='{topic}'.")
        logger_(f"-> advice={len(all_adv)}, mapping={len(all_map)}, context={len(all_ctx)}")

def generate_merge_dedup_tables(
    topic:str,
    assets_dir:str,
    n_batches=5,
    n_per_batch=20,
    advice_context: str | None = None,
    output_dir="outputs",
    verbose=False
):
    """
    Execute the full multi-batch generation pipeline with filtering and renumbering.

    This function serves as the backbone of the CSV preparation process. It calls
    `generate_all_csv_batches` to produce multiple batches of Advice, Context, and
    Mapping files, then applies sequential cleaning steps:
    1. Merge all per-batch CSVs into single datasets.
    2. Filter out any invalid or incomplete rows across all three CSVs.
    3. Renumber the remaining rows to maintain synchronized and contiguous `Num` values.
    4. Optionally audit the resulting files for structural integrity.

    The resulting synchronized CSVs are ready for narrative text generation.

    Parameters
    ----------
    client : openai.OpenAI
        Authenticated OpenAI client for all model-based generations.
    topic : str
        Human-readable topic name used for naming files and directories.
    assets_dir : str
        Directory containing the official SN.csv, DE.csv, style.txt, and examples.txt.
    n_batches : int, optional
        Number of batches to generate (default 5).
    n_per_batch : int, optional
        Number of rows per batch (default 20).
    output_dir : str, optional
        Main output directory where all generated and filtered CSVs will be stored.
    model_advice : str, optional
        Model used for generating Advice (default 'gpt-4o-mini').
    model_mapping : str, optional
        Model used for Mapping (default 'o3').
    model_context : str, optional
        Model used for Context (default 'gpt-4o-mini').
    verbose : bool, optional
        If True, prints progress messages and summary reports.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing paths to the final synchronized CSVs:
        (advice_path, mapping_path, context_path).

    Notes
    -----
    - Each CSV returned corresponds to the *FilteredRenumeroted* stage of the pipeline.
    - This function ensures structural and semantic consistency across all
    data before narrative generation begins.
    - It should be invoked before calling `run_pipeline` or `generate_narratives`.
    """

    if verbose:
        logger_("\n=== PIPELINE MULTI-BATCH NARREMGEN ===")
        logger_(f"Topic            : {topic}")
        logger_(f"Directory output : {os.path.abspath(output_dir)}")
        logger_(f"Batches          : {n_batches} x {n_per_batch} rows each")


    generate_all_csv_batches(
        topic=topic,
        assets_dir=assets_dir,
        n_batches=n_batches,
        n_per_batch=n_per_batch,
        advice_context=advice_context,
        output_dir=output_dir,
        verbose=verbose
    )

    merge_and_filter(topic=topic, output_dir=output_dir, verbose=verbose)
    renumerote_filtered(topic=topic, output_dir=output_dir, verbose=verbose)
    if verbose: audit_filtered(topic=topic, output_dir=output_dir)

    advice_path  = os.path.join(output_dir, build_csv_name("advice", "FilteredRenumeroted", topic))
    mapping_path = os.path.join(output_dir, build_csv_name("mapping", "FilteredRenumeroted", topic))
    context_path = os.path.join(output_dir, build_csv_name("context", "FilteredRenumeroted", topic))

    if verbose:
        logger_("\nPipeline multi-batch finished (re-numbered files ready).")
        logger_(f"Advice file found : {advice_path}")
        logger_(f"Mapping file found: {mapping_path}")
        logger_(f"Context file found: {context_path}")

    return advice_path, mapping_path, context_path
