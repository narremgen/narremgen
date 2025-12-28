from __future__ import annotations

"""
narremgen.main
==============
CLI entrypoint for Narremgen pipelines. Recommended (once packaged/installed):
    python -m narremgen.main --topic "Walking in the city"
    # optional: bypass pre-gen CSVs (must be FilteredRenumeroted)
    # python -m narremgen.main --topic "Walking in the city" \
    #   --bypass-advice-csv /path/Advice_FilteredRenumeroted_*.csv \
    #   --bypass-mapping-csv /path/Mapping_FilteredRenumeroted_*.csv \
    #   --bypass-context-csv /path/Context_FilteredRenumeroted_*.csv
Design goals:
- Default output directory is writable (~/.narremgen/outputs).
- Assets can be loaded from packaged resources (*/settings) or via --assets-dir.
- Optional pre-gen bypass: accept externally edited FilteredRenumeroted CSVs (Advice/Mapping/Context),
  copy them into the new workdir, run checks, then continue narrative generation as usual. 
- Variants step supports resume/fill-holes by default (overwrite only with --overwrite-existing).
- Logging always goes to workdir/pipeline.log; console logging is opt-in.
- API keys are read from environment variables first, then optional local files (key*.txt).
- API key quick checks (Windows-safe):
  - python -c "import os; print(repr(os.getenv('OPENAI_API_KEY')))"
  - PowerShell: echo $env:OPENAI_API_KEY
  - User Environment Variables from Settings in System Configuration Panel
  If env vars are unreliable, prefer a key file containing ONLY the raw key (one line as sk-...).
"""

import argparse
import importlib.resources as resources
import logging
import os
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from importlib.metadata import packages_distributions
import pandas as pd

from .analyzestats import build_global_stats_table
from .llmcore import LLMConnect
from .pipeline import run_pipeline
from .themes import run_llm_theme_pipeline
from .utils import build_csv_name, get_workdir_for_topic
from .utils import find_last_workdir, slugify_topic
from .variants import run_one_variant_pipeline
from .chapters import create_chapters_from_theme_assignments
from .export import build_merged_tex_from_csv
from .utils import load_neutral_data

DEFAULT_NEUTRAL_BATCHES = 2
DEFAULT_NEUTRAL_PER_BATCH = 20
DEFAULT_VARIANT_BATCH_SIZE = 20
DEFAULT_VARIANT_MAX_TOKENS = 3000
DEFAULT_THEMES_MIN = 7
DEFAULT_THEMES_MAX = 12
DEFAULT_THEMES_BATCH_SIZE = 20
THEMES_JSON_NAME = "advice_groups_llm.json"
THEMES_ASSIGNMENT_JSON_NAME = "advice2groups_llm.json"
DEFAULT_OUTPUT_ROOT = Path.home() / ".narremgen" / "outputs"
DEFAULT_LLM_MODELS = {
    "ADVICE": "openai\\gpt-4o-mini",
    "MAPPING": "openai\\gpt-4.1",
    "CONTEXT": "openai\\gpt-4o-mini",
    "NARRATIVE": "openai\\gpt-4o-mini",
    "THEME_ANALYSIS": "openai\\gpt-4.1-mini",
    "VARIANTS_GENERATION": "openai\\gpt-4o-mini",
}
MODEL_ENV_PREFIX = "NARREMGEN_MODEL_"
MODEL_ARG_BINDINGS: Dict[str, tuple[str, str, str]] = {
    "ADVICE": ("model_advice", "--model-advice", f"{MODEL_ENV_PREFIX}ADVICE"),
    "MAPPING": ("model_mapping", "--model-mapping", f"{MODEL_ENV_PREFIX}MAPPING"),
    "CONTEXT": ("model_context", "--model-context", f"{MODEL_ENV_PREFIX}CONTEXT"),
    "NARRATIVE": ("model_narrative", "--model-narrative", f"{MODEL_ENV_PREFIX}NARRATIVE"),
    "THEME_ANALYSIS": ("model_theme_analysis", "--model-theme-analysis", f"{MODEL_ENV_PREFIX}THEME_ANALYSIS"),
    "VARIANTS_GENERATION": ("model_variants_generation", "--model-variants-generation", f"{MODEL_ENV_PREFIX}VARIANTS_GENERATION"),
}
DEFAULT_MODEL_ENV_VAR = f"{MODEL_ENV_PREFIX}DEFAULT"
DEFAULT_MODEL_FLAG = "--default-model"
KEYFILE_BASENAME = "llmkey_narremgen.txt"


def _pkg_version() -> str:
    dists = packages_distributions().get("narremgen") or []
    if dists:
        try:
            return dist_version(dists[0])
        except PackageNotFoundError:
            pass
    try:
        return dist_version("narremgen")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def _split_provider(model_spec: str) -> tuple[str, str]:
    s = (model_spec or "").strip()
    if not s:
        return ("", "")
    for sep in ("\\", "/", ":"):
        if sep in s:
            p, m = s.split(sep, 1)
            return (p.strip().lower(), m.strip())
    return ("", s)

def _logger() -> logging.Logger:
    logger = logging.getLogger("narremgen.main")
    return logger


def _configure_logging(workdir: Path, verbose: bool, log_to_console: bool) -> Path:
    """
    Configure root logging:
    - Always file: workdir/pipeline.log
    - Optional console: --log-to-console
    """
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_path = workdir / "pipeline.log"
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    return log_path

def _resolve_assets_dir(user_assets_dir: Optional[str], stack: ExitStack) -> Path:
    """
    Priority:
      1) --assets-dir
      2) packaged resources: <package>/settings (tries current package then top package)
      3) dev fallback: <this_file_dir>/settings
    """
    here = Path(__file__).resolve().parent

    if user_assets_dir:
        path = Path(user_assets_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    pkg_full = __package__ or "narremgen"
    candidates = [pkg_full, pkg_full.split(".", 1)[0]]
    for pkg in candidates:
        try:
            res = resources.files(pkg).joinpath("settings")
            return stack.enter_context(resources.as_file(res))
        except Exception:
            continue

    dev_path = (here / "settings").resolve()
    if dev_path.exists():
        return dev_path

    raise RuntimeError(
        "Unable to locate packaged assets (narremgen/settings). Install the package or pass --assets-dir."
    )

def _check_assets_dir(assets_dir: Path) -> bool:
    required = [
        assets_dir / "DE.csv",
        assets_dir / "SN.csv",
        assets_dir / "SN_extended_raw.csv",
        assets_dir / "style.txt",
        assets_dir / "examples.txt",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        _logger().error("assets_dir incomplete (%s). Missing: %s", assets_dir, ", ".join(missing))
        return False
    return True

def _format_models_map(d: dict) -> str:
    items = sorted((str(k), str(v)) for k, v in (d or {}).items())
    return "\n".join([f"  - {k}: {v}" for k, v in items]) or "  (empty)"


def _init_llm(llm_models: Dict[str, str], default_model: str, max_tokens: int,
              temperature: float, request_timeout: int, verbose: bool,
              diagnostic_dry_run: bool= False) -> dict[str, str] | None:
    if llm_models is None: 
        llm_models=DEFAULT_LLM_MODELS
    LLMConnect.init_global(
        llmodels=llm_models,
        default_model=default_model,
        temperature=temperature,
        max_tokens=max(256, int(max_tokens)), #max_tokens=1024,
        request_timeout=request_timeout,
    )
    llm = LLMConnect.get_global()
    if diagnostic_dry_run:
        try:
            return llm.auto_dry_run(verbose=verbose)
        except Exception as e:
            log = _logger()
            log.debug("LLM dry-run failed:\n%s", traceback.format_exc())
            mapping_txt = _format_models_map(llm_models)
            log.warning("LLM auto_dry_run failed. Model mapping:\n%s", mapping_txt)
            log.warning("Default model     : %s", default_model)
            log.warning("Did you set the API key / plugin config for all models?")
            return {"__diagnostic__": f"FAILED (exception)"}
    return None

def _build_algebra_paths(assets_dir: Path, workdir: Path, topic_slug: str) -> Dict[str, Path]:
    return {
        "sn_pathdf": assets_dir / "SN_extended_raw.csv",
        "de_pathdf": assets_dir / "DE.csv",
        "ki_pathdf": workdir / build_csv_name("context", "FilteredRenumeroted", topic_slug),
        "opstc_pathdf": assets_dir / "operators_structural_raw.csv",
        "opstl_pathdf": assets_dir / "operators_stylistic_raw.csv",
    }


def _check_algebra_paths(paths: Dict[str, Path]) -> bool:
    missing = [str(p) for p in paths.values() if not Path(p).exists()]
    if missing:
        _logger().error("Missing algebra inputs: %s", "; ".join(missing))
        return False
    return True


def _default_variant_registry(assets_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "direct": {
            "name": "direct",
            "title": "Direct",                        
            "prompt_path": str((assets_dir / "prompt_direct_variant_example.txt").resolve()),
            "use_sn": True,
            "use_de": True,
            "use_k": True,
            "use_ops": False,
        },
        "formal": {
            "name": "formal",
            "title": "Formal",
            "prompt_path": str((assets_dir / "prompt_formal_variant_example.txt").resolve()),          
            "use_sn": False,
            "use_de": False,
            "use_k": False,
            "use_ops": False,
        },
    }


def _required_env_for_provider(provider: str) -> list[list[str]]:
    """
    Return required environment variable keys for a provider.

    The function maps provider names to the environment variable(s) that must be set
    for authentication. Some providers accept multiple possible keys (e.g., an alias
    or a fallback), which is represented as a list of alternatives.

    Parameters
    ----------
    provider : str
        Provider identifier (for example: "openai", "gemini", "openrouter", "mistral").

    Returns
    -------
    list[str] | list[list[str]]
        One of:
        - `["ENV_VAR"]` meaning a single required key,
        - `["ENV_VAR1", "ENV_VAR2"]` meaning any one of those keys is acceptable,
        - `[["ENV_VAR1", "ENV_VAR2"]]` meaning ENV_VAR1 OR ENV_VAR2 is required,
        depending on how the calling code interprets alternatives,
        - `[]` meaning: no key is required.
    """

    p = provider.lower().strip()
    if p == "openai":
        return [["OPENAI_API_KEY"]]
    if p == "mistral":
        return [["MISTRAL_API_KEY"]]
    if p == "openrouter":
        return [["OPENROUTER_API_KEY", "OPENAI_API_KEY"]]
    if p in ("grok", "xai"):
        return [["XAI_API_KEY", "GROK_API_KEY"]]
    if p == "huggingface":
        return [["HUGGINGFACE_API_KEY"]]
    if p == "ollama":
        return [] 
    return [] 

def _preflight_env_keys(llm_models: dict[str, str]) -> None:
    roles_provider = _providers_from_llm_models(llm_models)

    prov_to_roles: dict[str, list[str]] = {}
    for role, prov in roles_provider.items():
        prov_to_roles.setdefault(prov, []).append(role)

    def _satisfies(req: list[list[str]]) -> bool:        
        return any(any(os.getenv(k) for k in group) for group in (req or []))

    missing: list[tuple[str, list[list[str]], list[str]]] = []
    for prov, roles in sorted(prov_to_roles.items()):
        req = _required_env_for_provider(prov)
        if req and not _satisfies(req):
            missing.append((prov, req, roles))

    if not missing:
        return

    log = _logger()
    log.error("Missing API keys (env) for configured llm_models:")
    for prov, req, roles in missing:
        pretty = " OR ".join(" / ".join(group) for group in req)
        log.error("  - provider=%s roles=[%s] needs: %s", prov, ", ".join(sorted(roles)), pretty)

    raise SystemExit(2)


def _detect_cli_flags(argv: List[str]) -> Set[str]:
    return {token.split("=", 1)[0] for token in argv if token.startswith("--")}


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_llm_models(args: argparse.Namespace, cli_flags: Set[str]) -> tuple[Dict[str, str], str, Dict[str, str]]:
    models = DEFAULT_LLM_MODELS.copy()
    env_overrides: Dict[str, str] = {}

    for role, (attr_name, flag_name, env_name) in MODEL_ARG_BINDINGS.items():
        env_value = _strip_or_none(os.getenv(env_name))
        if env_value and flag_name not in cli_flags:
            models[role] = env_value
            env_overrides[env_name] = env_value
            continue

        arg_value = _strip_or_none(getattr(args, attr_name, None))
        models[role] = arg_value or DEFAULT_LLM_MODELS[role]

    default_model = models.get("NARRATIVE", DEFAULT_LLM_MODELS["NARRATIVE"])
    env_default = _strip_or_none(os.getenv(DEFAULT_MODEL_ENV_VAR))
    if env_default and DEFAULT_MODEL_FLAG not in cli_flags:
        default_model = env_default
        env_overrides[DEFAULT_MODEL_ENV_VAR] = env_default
    else:
        arg_default = _strip_or_none(getattr(args, "default_model", None))
        default_model = arg_default or default_model

    return models, default_model, env_overrides


def _iter_keyfile_candidates() -> List[Path]:
    here = Path(__file__).resolve().parent
    return [
        Path.cwd() / KEYFILE_BASENAME,
        here / KEYFILE_BASENAME,
    ]

def _read_keyfile_entries(explicit_path: Optional[Path] = None) -> Tuple[Dict[str, str], Optional[str], Optional[Path]]:
    """
    Multi-pass keyfile reader.

    Attempts to interpret the keyfile content in two ways:
    1. Explicit structure (recommended): Looks for lines in `VAR=VALUE` or `VAR:VALUE` format.
    2. Raw content (fallback): If no key/value structure is found, treats the first
       non-empty line as a raw key string.

    Args:
        explicit_path: Optional path to a specific file. If None, checks default candidates.

    Returns:
        Tuple containing:
        - entries (Dict): Dictionary of explicit environment variables found.
        - raw_key (Optional[str]): The raw key string if no explicit entries were found.
        - source (Optional[Path]): The path of the file used.
    """
    log = _logger()

    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    else:
        candidates.extend(_iter_keyfile_candidates())

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue

            content = candidate.read_text(encoding="utf-8", errors="strict")
            entries: Dict[str, str] = {}
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
            
            for line in lines:
                sep = None
                if "=" in line:
                    sep = "="
                elif ":" in line:
                    sep = ":" 

                if sep:
                    k, v = line.split(sep, 1)
                    k = k.strip().upper()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        entries[k] = v
            
            if entries:
                return entries, None, candidate

            if lines:
                return {}, lines[0], candidate

        except Exception:
            log.debug("Failed to read %s:\n%s", candidate, traceback.format_exc())

    return {}, None, None


def _apply_keyfile_api_keys(
    llm_models: Dict[str, str],
    explicit_keyfile: Optional[Path] = None,
) -> Tuple[Set[str], Optional[Path]]:
    """
    Apply API keys from a keyfile into environment variables.

    Behavior
    --------
    - If `explicit_keyfile` is provided: read it and overwrite any corresponding
    environment variables found in the file.
    - Only environment variables relevant to active/configured providers are applied.
    - If the file contains a single raw key line (for example: `sk-example-123456`),
    the function infers which provider to apply by inspecting `llm_models`. If
    exactly one active provider is present, that provider receives the key.

    Parameters
    ----------
    llm_models : dict[str, str]
        Provider-to-model mapping for the current run.
    explicit_keyfile : pathlib.Path | None, default None
        Path to a keyfile to apply.

    Returns
    -------
    tuple[set[str], pathlib.Path | None]
        A tuple `(applied_keys, keyfile_path)` where `applied_keys` is the set of
        environment variable names written, and `keyfile_path` is the resolved file
        path used (if any).
    """

    entries, raw_key, source = _read_keyfile_entries(explicit_path=explicit_keyfile)
    
    if explicit_keyfile and not entries and not raw_key:
        raise FileNotFoundError(f"API key file not found or empty: {explicit_keyfile}")

    if not entries and not raw_key:
        return set(), None

    overridden_vars: Set[str] = set()

    if entries:
        #providers = _providers_from_llm_models(llm_models)
        for k, v in entries.items():
            os.environ[k] = v
            overridden_vars.add(k)
        return overridden_vars, source

    if raw_key:
        active_providers = set(_providers_from_llm_models(llm_models).values())
        needed = []
        for prov in active_providers:
            reqs = _required_env_for_provider(prov)
            if reqs:
                needed.append((prov, reqs[0][0]))

        if len(needed) == 0:
             _logger().warning("Raw key provided but no active provider needs one. Ignored.")
             return set(), source

        if len(needed) == 1:
            prov_name, target_var = needed[0]
            os.environ[target_var] = raw_key
            overridden_vars.add(target_var)
            _logger().info(f"[KEYFILE] Auto-assigned raw key to {target_var} (for {prov_name})")
            return overridden_vars, source
        
        else:
            conflicts = [n[0] for n in needed]
            raise ValueError(
                f"Ambiguous raw key. You utilize multiple providers needing keys: {conflicts}. "
                f"Please use 'VAR_NAME=KEY' format in your key file."
            )

    return set(), None

def _providers_from_llm_models(llm_models: Dict[str, str]) -> Dict[str, str]:
    """
    Extract the LLM provider name for each logical role from a model mapping.

    The input dictionary maps logical roles (e.g. "ADVICE", "NARRATIVE") to
    model specifications of the form "provider\\model", "provider/model",
    or similar. This function parses each specification and returns a new
    dictionary mapping each role to its resolved provider name.

    Entries with empty or invalid model specifications are ignored.

    Parameters
    ----------
    llm_models : dict[str, str]
        Mapping from logical roles to provider-qualified model strings.

    Returns
    -------
    dict[str, str]
        Mapping from logical roles to provider names (lowercased).
    """
    providers: Dict[str, str] = {}
    for role, spec in llm_models.items():
        if spec:
            provider, _ = _split_provider(spec)
            if provider:
                providers[role] = provider
    return providers


def _parse_args() -> argparse.Namespace:
    """
    Parse and validate all command-line options supported by the Narremgen CLI.

    This function defines the complete public CLI surface of the application.
    All arguments are grouped conceptually into the following categories:

    1) Core execution
       --topic
           Optional here, but required for normal runs (except diagnostics).
           High-level topic used to generate the narrative pipeline.
           The topic is slugified and used to name output directories.

       --version
           Print package version and exit.

    2) Assets and filesystem layout
       --assets-dir
           Path to a directory containing the required assets (CSV, prompts, styles).
           Overrides packaged resources. Intended for development or custom setups.

       --output-dir
           Root directory where per-topic workdirs are created.
           Defaults to ~/.narremgen/outputs.

       --workdir
           Explicit work directory. If provided, --output-dir and automatic
           workdir creation logic are bypassed entirely.

    3) API keys and environment handling
       --api-key-file
           Path to a file containing API keys.
           Supported formats:
             - Explicit ENV_VAR=VALUE or ENV_VAR:VALUE entries (recommended).
             - A single raw key line (e.g. sk-[etc]), auto-assigned if unambiguous.

       --no-preflight-env
           Disable environment preflight checks for required provider keys.
           Intended for advanced or controlled execution environments.

    4) Pipeline stage control
       --skip-neutral
           Skip the neutral (base) generation stage.

       --skip-variants
           Skip variants generation.

       --skip-themes
           Skip LLM-based thematic classification.

       --skip-stats
           Skip computation of global statistics across variants.

    5) Pre-generation bypass (advanced)
       --bypass-advice-csv
       --bypass-mapping-csv
       --bypass-context-csv
           Provide externally edited FilteredRenumeroted CSVs to bypass the
           corresponding neutral-generation steps. All three must be provided
           together or execution aborts.

    6) Neutral generation parameters
       --batches
           Number of neutral-generation batches.

       --per-batch
           Number of items per batch.

       --dialogue-mode
           Dialogue formatting mode: none, single, or short.

       --output-format
           Output format for neutral text generation (txt or tex).

       --extra-instructions
           Free-form string appended to neutral-generation prompts.

    7) Variants generation parameters
       --variants
           List of variant names to run. Defaults to all registered variants.

       --variant-batch-size
           Batch size for variant generation.

       --variant-max-tokens
           Maximum tokens per variant generation call.

       --overwrite-existing
           Overwrite existing variant outputs instead of resuming/filling gaps.

       --do-stats
           Enable text statistics computation during variant generation.

       --use-emo
           Add emotion/sentiment metrics to text statistics (optional heavy dependencies).
           Requires --do-stats.
         
    8) Theme analysis parameters
       --themes-min
       --themes-max
           Minimum and maximum number of themes to extract.

       --themes-batch-size
           Batch size for theme classification.

       --themes-retries
           Maximum retry attempts for the themes step.

    9) LLM configuration
       --default-model
           Fallback provider\\model used when no role-specific mapping exists.

       --model-advice
       --model-mapping
       --model-context
       --model-narrative
       --model-theme-analysis
       --model-variants-generation
           Role-specific provider\\model overrides.

       Environment-variable equivalents are supported and overridden by CLI flags.

    10) Logging and verbosity
       --verbose
           Enable DEBUG-level logging to file.

       --log-to-console
           Also emit logs to stdout.

       --quiet
           Suppress most console output while keeping file logs.

    11) Dry connection test for LLM models
        --diagnostic-dry-run
            Print the resulting test to connect the involved models and close.
        # python -m narremgen --diagnostic-dry-run --model-advice mistral\mistral-tiny                              |TESTED
        # python -m narremgen --diagnostic-dry-run --model-advice ollama\\phi3-chat:latest                          |TESTED
        # python -m narremgen --diagnostic-dry-run --model-advice openrouter\meta-llama/llama-3.3-70b-instruct:free |TESTED
        # python -m narremgen --diagnostic-dry-run --model-advice "openrouter\google/gemini-2.0-flash-exp:free"     |TESTED
        # python -m narremgen --diagnostic-dry-run --model-advice "gemini\gemini-2.0-flash"                         |TESTED

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with all defaults applied.
    """
    p = argparse.ArgumentParser(
        prog="narremgen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
   
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_pkg_version()}",
    )

    p.add_argument("--topic", required=False, help="Topic, e.g. 'Walking in the city'")
    p.add_argument("--assets-dir", default=None, help="Path to settings/assets directory")

    p.add_argument(
        "--api-key-file",
        default=None,
        help="Path to API key file. Supports multi-line ENV_VAR=VALUE (recommended), "
             "or single raw OpenAI key (sk-[etc]) for only one large language model."
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=f"Where to create outputs/<topic_slug>_k workdirs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    p.add_argument(
        "--workdir",
        default=None,
        help="Use an explicit workdir path (advanced). If provided, --output-dir/create_new are ignored.",
    )
    p.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Reuse an existing <topic_slug>_<k> workdir (requires --skip-neutral).",
    )
    p.add_argument(
        "--no-entry-numbers",
        action="store_true",
        help="Hide per-entry numbering in TeX/PDF exports.",
    )
    p.add_argument(
        "--no-snde",
        action="store_true",
        help="Hide SN/DE codes in TeX/PDF exports.",
    )
    p.add_argument("--export-book-tex", action="store_true", help="Export TeX books into variants/ directories (post-step).")
    p.add_argument("--merge-small-chapters", type=int, default=0, help="Merge chapters with <K items into 'Other' (0 disables).")
    p.add_argument("--chapters-json", default=None, help="Optional: explicit chapters JSON path; otherwise use workdir/themes/chapters_llm.json if present.")
    p.add_argument("--skip-neutral", action="store_true", help="Skip neutral generation (reuse existing workdir)")
    p.add_argument("--skip-variants", action="store_true", help="Skip variants generation")
    p.add_argument("--skip-themes", action="store_true", help="Skip LLM themes step")
    p.add_argument("--skip-stats", action="store_true", help="Skip global stats table")
    p.add_argument("--bypass-advice-csv", default=None, help="Bypass pre-gen: path to Advice_FilteredRenumeroted CSV")
    p.add_argument("--bypass-mapping-csv", default=None, help="Bypass pre-gen: path to Mapping_FilteredRenumeroted CSV")
    p.add_argument("--bypass-context-csv", default=None, help="Bypass pre-gen: path to Context_FilteredRenumeroted CSV")
    p.add_argument("--batches", type=int, default=DEFAULT_NEUTRAL_BATCHES, help="Neutral: number of batches")
    p.add_argument("--per-batch", type=int, default=DEFAULT_NEUTRAL_PER_BATCH, help="Neutral: items per batch")
    p.add_argument("--dialogue-mode", default="single", choices=["none", "single", "short"], help="Neutral: dialogue mode")
    p.add_argument("--output-format", default="txt", choices=["txt", "tex"], help="Neutral: output format")
    p.add_argument("--advice-context", default="", help="Neutral: epistemic context injected only into advice generation (string)")
    p.add_argument("--advice-context-path", default=None, help="Path to a text file whose content is appended to advice context")
    p.add_argument("--extra-instructions", default=" ", help="Neutral: extra instructions appended to prompts (string)")
    p.add_argument("--extra-instructions-path",default=None,help="Path to a text file whose content is appended to prompts")
    p.add_argument("--variant-batch-size", type=int, default=DEFAULT_VARIANT_BATCH_SIZE, help="Variants: batch size")
    p.add_argument("--variant-max-tokens", type=int, default=DEFAULT_VARIANT_MAX_TOKENS, help="Variants: max tokens")
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Variants: overwrite existing CSVs (default: resume/fill holes)",
    )
    p.add_argument("--variants", nargs="*", default=None, help="Variant names to run (e.g. direct formal)")
    p.add_argument("--themes-min", type=int, default=DEFAULT_THEMES_MIN)
    p.add_argument("--themes-max", type=int, default=DEFAULT_THEMES_MAX)
    p.add_argument("--themes-batch-size", type=int, default=DEFAULT_THEMES_BATCH_SIZE)
    p.add_argument("--themes-retries", type=int, default=5, help="Themes: max retries")
    p.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    p.add_argument("--request-timeout", type=int, default=600, help="LLM request timeout (seconds)")
    p.add_argument("--default-model", default=DEFAULT_LLM_MODELS["NARRATIVE"], help=f"LLM: fallback model (env: {DEFAULT_MODEL_ENV_VAR})")
    p.add_argument("--model-advice", default=DEFAULT_LLM_MODELS["ADVICE"], help=f"LLM: model for ADVICE (env: {MODEL_ARG_BINDINGS['ADVICE'][2]})")
    p.add_argument("--model-mapping", default=DEFAULT_LLM_MODELS["MAPPING"], help=f"LLM: model for MAPPING (env: {MODEL_ARG_BINDINGS['MAPPING'][2]})")
    p.add_argument("--model-context", default=DEFAULT_LLM_MODELS["CONTEXT"], help=f"LLM: model for CONTEXT (env: {MODEL_ARG_BINDINGS['CONTEXT'][2]})")
    p.add_argument("--model-narrative", default=DEFAULT_LLM_MODELS["NARRATIVE"], help=f"LLM: model for NARRATIVE (env: {MODEL_ARG_BINDINGS['NARRATIVE'][2]})")
    p.add_argument("--model-theme-analysis", default=DEFAULT_LLM_MODELS["THEME_ANALYSIS"], help=f"LLM: model for THEME_ANALYSIS (env: {MODEL_ARG_BINDINGS['THEME_ANALYSIS'][2]})")
    p.add_argument("--model-variants-generation", default=DEFAULT_LLM_MODELS["VARIANTS_GENERATION"], help=f"LLM: model for VARIANTS_GENERATION (env: {MODEL_ARG_BINDINGS['VARIANTS_GENERATION'][2]})")

    p.add_argument("--do-stats", action="store_true",
                help="Compute text statistics (neutral + variants).")
    p.add_argument("--do-add-emo-to-stat", action="store_true",
                help="Add emotion/sentiment analysis to text stats (slow; heavy deps).")
    p.add_argument("--do-plots", action="store_true",
                help="Generate analysis plots (requires matplotlib).")

    p.add_argument("--verbose", action="store_true", help="Verbose logging to file (DEBUG)")
    p.add_argument("--log-to-console", action="store_true", help="Also log to console (stdout)")
    p.add_argument("-q", "--quiet", action="store_true", help="Less console output (still logs to file)")
 
    p.add_argument("--no-preflight-env", action="store_true", help="Skip env API-key preflight checks")

    p.add_argument(
        "--diagnostic-dry-run",
        action="store_true",
        help=(
            "Run an explicit LLM connectivity dry-run at startup. "
            "This performs real network calls to configured providers and may be slow. "
            "Intended for diagnostics only. Disabled by default."
        ),
    )

    return p.parse_args()


def main() -> int:
    """
    Execute the Narremgen command-line pipeline.

    This function is the authoritative runtime entry point of the application.
    It coordinates configuration resolution, environment validation, LLM
    initialization, and sequential execution of all enabled pipeline stages.

    High-level responsibilities:
    - Resolve assets and working directories deterministically.
    - Configure structured logging (file-first, console optional).
    - Resolve LLM model mappings from defaults, environment variables, and CLI.
    - Apply API keys from an optional key file, with controlled environment mutation.
    - Perform provider-level preflight checks unless explicitly disabled.
    - Initialize the global LLM router.
    - Execute pipeline stages conditionally:
        * neutral generation
        * variants generation
        * thematic classification
        * global statistics aggregation
    - Enforce failure isolation between stages and provide meaningful exit codes.

    Design constraints:
    - All filesystem side effects are confined to a single work directory.
    - Environment variables are only modified when a key file is explicitly provided.
    - Partial execution and resumption are supported without implicit overwrites.
    - The function must be safe to invoke from a packaged installation or a dev tree.

    Exit codes:
    - 0 : Successful execution.
    - 1 : Runtime failure during pipeline execution.
    - 2 : Configuration, argument, or preflight validation error.

    Returns
    -------
    int
        Process exit status suitable for use with sys.exit().
    """
    args = _parse_args()
    log = _logger()

    if args.workdir and (args.run_id is not None or args.output_dir):
        print("[FATAL] --workdir is mutually exclusive with --run-id/--output-dir.")
        return 2

    needs_llm = args.diagnostic_dry_run or (not args.skip_neutral) \
                  or (not args.skip_variants) or (not args.skip_themes)
    if needs_llm:
        cli_flags = _detect_cli_flags(sys.argv[1:])
        llm_models, default_model, env_overrides = _resolve_llm_models(args, cli_flags)
        keyfile_arg = _strip_or_none(getattr(args, "api_key_file", None))
        keyfile_path = Path(keyfile_arg).expanduser().resolve() if keyfile_arg else None
        keyfile_env_applied: Set[str] = set()
        if keyfile_path:
            applied, src = _apply_keyfile_api_keys(llm_models, explicit_keyfile=keyfile_path)
            keyfile_env_applied = set(applied or [])
            msg =    f"[KEYFILE] applied to env vars: {', '.join(sorted(keyfile_env_applied))} (from {src})"
            if args.diagnostic_dry_run:
                if not args.quiet: print(msg)
            else:
                log.info(msg) 

        if not bool(args.no_preflight_env):
            _preflight_env_keys(llm_models)

        if args.verbose:
            for role, spec in llm_models.items():
                prov, _ = _split_provider(spec)
                if not prov:
                    continue
                req = _required_env_for_provider(prov)
                if not req:
                    continue
                seen = {k: bool(os.getenv(k)) for group in req for k in group}
                log.debug("ENV for %s (%s): %s", role, prov, seen)

        if bool(args.verbose):
            log.info("LLM model mapping:\n%s", _format_models_map(llm_models))
            log.info("LLM default model: %s", default_model)
            if env_overrides:
                env_pairs = ", ".join(f"{k}={v}" for k, v in sorted(env_overrides.items()))
                log.info("LLM env overrides: %s", env_pairs)
            if keyfile_env_applied:
                vars_txt = ", ".join(sorted(keyfile_env_applied))
                log.info("LLM keyfile applied to env vars: %s", vars_txt)

        results = _init_llm(
            llm_models=llm_models,
            default_model=default_model,
            max_tokens=int(max(args.variant_max_tokens, 1024)),
            temperature=float(args.temperature),
            request_timeout=int(args.request_timeout),
            verbose=bool(args.verbose),
            diagnostic_dry_run=bool(args.diagnostic_dry_run),
        )
        if args.diagnostic_dry_run:
            if results is None:
                results = {"__diagnostic__": "FAILED (no results)"}
            failed = {m:s for m,s in results.items() if s.startswith("FAILED")}
            skipped = {m:s for m,s in results.items() if s.startswith("SKIPPED")}

            if not args.quiet:
                for m, s in sorted(results.items()):
                    print(f"[DIAG] {m}: {s}")
                if failed:
                    print(f"[DIAG] FAILED: {len(failed)} / {len(results)}")
                else:
                    print(f"[DIAG] OK: {len(results) - len(skipped)} tested, {len(skipped)} skipped")

            return 2 if failed else 0

    if not args.topic or not args.topic.strip():
        print("[FATAL] --topic is required unless --diagnostic-dry-run is used")
        return 2

    topic = args.topic.strip()
    topic_slug = slugify_topic(topic)

    stack = ExitStack()
    try:
        assets_arg = _strip_or_none(args.assets_dir)
        try:
            assets_dir = _resolve_assets_dir(assets_arg, stack)
            if not _check_assets_dir(assets_dir):
                if not args.quiet:
                    print("[FATAL] invalid assets_dir.")
                return 2
        except FileNotFoundError as exc:
            print(f"[FATAL] assets directory not found: {exc}. Pass --assets-dir=PATH.")
            return 2
        except RuntimeError as exc:
            print(f"[FATAL] {exc} (use --assets-dir to provide a path).")
            return 2
    
        if args.workdir:
            workdir = Path(args.workdir).expanduser().resolve()
            if args.skip_neutral and not workdir.is_dir():
                print(f"[FATAL] Workdir not found: {workdir}")
                return 2
            workdir.mkdir(parents=True, exist_ok=True)
        else:
            if args.output_dir:
                output_dir = Path(args.output_dir).expanduser().resolve()
            else:
                output_dir = DEFAULT_OUTPUT_ROOT
            output_dir.mkdir(parents=True, exist_ok=True)

            if args.run_id is not None:
                if not args.skip_neutral:
                    print("[FATAL] --run-id requires --skip-neutral (or use --workdir).")
                    return 2
                workdir = output_dir / f"{topic_slug}_{int(args.run_id)}"
                if not workdir.is_dir():
                    print(f"[FATAL] Workdir not found: {workdir}")
                    return 2
            elif args.skip_neutral:
                workdir = find_last_workdir(output_dir, topic)
            else:
                workdir = get_workdir_for_topic(output_dir, topic_slug, create_new=True)

        log_path = _configure_logging(
            workdir,
            verbose=bool(args.verbose),
            log_to_console=bool(args.log_to_console) and not bool(args.quiet),
        )      

        if not args.quiet:
            print(f"[INFO] WORKDIR = {workdir}")
            print(f"[INFO] LOGFILE = {log_path}")

        if not args.skip_neutral:
            if not args.quiet:
                print("[STEP] Neutral generation…")
            try:
                csv_for_bypass_pre_gen = None
                if args.bypass_advice_csv or args.bypass_mapping_csv or args.bypass_context_csv:
                    if not (args.bypass_advice_csv and args.bypass_mapping_csv and args.bypass_context_csv):
                        raise ValueError(
                            "Bypass pre-gen requires all three: "
                            "--bypass-advice-csv, --bypass-mapping-csv, --bypass-context-csv"
                        )
                    csv_for_bypass_pre_gen = {
                        "advice_path": str(args.bypass_advice_csv),
                        "mapping_path": str(args.bypass_mapping_csv),
                        "context_path": str(args.bypass_context_csv),
                    }

                advice_context = (args.advice_context or "").strip()
                if args.advice_context_path:
                    p = Path(args.advice_context_path)
                    if not p.exists():
                        raise FileNotFoundError(f"Advice context file not found: {p}")
                    advice_context = (
                        advice_context + "\n\n" +
                        p.read_text(encoding="utf-8").strip()
                    ).strip()
                if not advice_context:
                    advice_context = None

                extra_instructions = args.extra_instructions or ""
                if args.extra_instructions_path:
                    p = Path(args.extra_instructions_path)
                    if not p.exists():
                        raise FileNotFoundError(f"Extra instructions file not found: {p}")
                    extra_instructions = (
                        extra_instructions + "\n\n" +
                        p.read_text(encoding="utf-8").strip()
                    ).strip()

                out_file = run_pipeline(
                    topic=topic,
                    workdir=str(workdir),
                    assets_dir=str(assets_dir),
                    csv_for_bypass_pre_gen=csv_for_bypass_pre_gen,
                    n_batches=int(args.batches),
                    n_per_batch=int(args.per_batch),
                    advice_context=advice_context,
                    output_format=str(args.output_format),
                    extra_instructions=extra_instructions,
                    dialogue_mode=str(args.dialogue_mode),
                    verbose=bool(args.verbose),
                )
                log.info("Neutral pipeline done. final=%s", out_file)
            except Exception:
                log.error("Neutral pipeline failed:\n%s", traceback.format_exc())
                if not args.quiet:
                    print("[FATAL] Neutral pipeline failed (see logfile).")
                return 1

        list_variant_data: List[Dict[str, Any]] = []
        if not args.skip_variants:
            if not args.quiet:
                print("[STEP] Variants generation…")

            registry = _default_variant_registry(assets_dir)
            requested = [v.strip() for v in (args.variants or []) if v.strip()] or list(registry.keys())

            for name in requested:
                if name not in registry:
                    log.warning("Unknown variant '%s' (available: %s)", name, ", ".join(sorted(registry.keys())))
                    continue
                vd = dict(registry[name])
                p = Path(vd["prompt_path"])
                if not p.is_absolute():
                    vd["prompt_path"] = str((assets_dir / p).resolve())
                list_variant_data.append(vd)

            if not list_variant_data:
                log.warning("No variants to run.")
            else:
                algebra_path = _build_algebra_paths(assets_dir=assets_dir, workdir=workdir, topic_slug=topic_slug)
                if not _check_algebra_paths(algebra_path):
                    if not args.quiet:
                        print("[FATAL] Missing algebra inputs (see logfile).")
                    return 2

                overwrite = bool(args.overwrite_existing)
                do_stats = bool(args.do_stats)
                use_emo  = bool(args.do_add_emo_to_stat) and do_stats
                do_plots = bool(args.do_plots) and do_stats

                for vd in list_variant_data:
                    name = vd["name"]
                    if not args.quiet:
                        print(f"[INFO] Variant {name} (overwrite={overwrite})")
                    try:
                        out_variant = run_one_variant_pipeline(
                            workdir=workdir,
                            topic=topic,
                            variant_data=vd,
                            algebra_path=algebra_path,
                            max_tokens=int(args.variant_max_tokens),
                            batch_size=int(args.variant_batch_size),
                            verbose=bool(args.verbose),
                            overwrite_existing=overwrite,
                            do_stats=do_stats,
                            use_emo=use_emo,
                            do_plots=do_plots,
                        )
                        log.info("Variant %s done: %s", name, out_variant)
                    except Exception:
                        log.error("Variant %s failed:\n%s", name, traceback.format_exc())
                        if not args.quiet:
                            print(f"[WARN] Variant {name} failed (see logfile).")

        if not args.skip_themes:
            if not args.quiet:
                print("[STEP] Themes classification…")
            attempt = 0
            max_retries = max(1, int(args.themes_retries))
            while attempt < max_retries:
                attempt += 1
                try:
                    run_llm_theme_pipeline(
                        workdir=workdir,
                        themes_json_name=THEMES_JSON_NAME,
                        assignments_json_name=THEMES_ASSIGNMENT_JSON_NAME,
                        n_themes_min=int(args.themes_min),
                        n_themes_max=int(args.themes_max),
                        batch_size=int(args.themes_batch_size),
                        verbose=bool(args.verbose),
                    )
                    log.info("Themes step success.")

                    assignments_path = workdir / "themes" / THEMES_ASSIGNMENT_JSON_NAME
                    chapters_path = workdir / "themes" / "chapters_llm.json"
                    assignments_path = workdir / "themes" / THEMES_ASSIGNMENT_JSON_NAME
                    if assignments_path.is_file():
                        create_chapters_from_theme_assignments(
                            assignments_json=assignments_path,
                            out_json=chapters_path,
                            merge_small_chapters=int(args.merge_small_chapters),
                            small_chapters_title="Other",
                        )
                    break
                except Exception:
                    log.error("Themes step failed (attempt %s/%s):\n%s", attempt, max_retries, traceback.format_exc())
                    if attempt >= max_retries:
                        if not args.quiet:
                            print("[WARN] Themes failed (max retries reached).")
                    else:
                        if not args.quiet:
                            print("[INFO] Themes retry…")

        if not args.skip_stats:
            if not args.quiet:
                print("[STEP] Global stats across variants…")

            stats_files: Dict[str, Path] = {}
            for vd in list_variant_data:
                vt = vd["name"].lower()
                stats_path = workdir / "variants" / vt / f"stats_neutral_vs_{vt}.csv"
                if stats_path.exists():
                    stats_files[vt] = stats_path
                else:
                    log.warning("Missing stats file for %s: %s", vt, stats_path)

            if not stats_files:
                log.warning("No stats files found; skipping global table.")
            else:
                stats_dict: Dict[str, pd.DataFrame] = {}
                for vt, path in stats_files.items():
                    stats_dict[vt] = pd.read_csv(path, sep=";")

                global_stats = build_global_stats_table(
                    stats_dict=stats_dict,
                    list_variants=list(stats_files.keys()),
                    metric_source="both",
                    verbose=bool(args.verbose),
                )

                if global_stats is not None:
                    out_csv = workdir / "stats_all_variants_table.csv"
                    global_stats.to_csv(out_csv, sep=";", index=True)
                    out_tex = workdir / "stats_all_variants_table.tex"
                    out_tex.write_text(global_stats.to_latex(index=True, escape=False), encoding="utf-8")
                    log.info("Global stats saved: %s ; %s", out_csv, out_tex)

        if args.export_book_tex:
            workdir_p = Path(workdir)
            try:
                load_neutral_data(workdir_p, verbose=bool(args.verbose))
            except FileNotFoundError as e:
                log.error("[TEX] Neutral CSV missing for TeX export")
                return 2

            chapters_json = None
            if getattr(args, "chapters_json", None):
                chapters_json = Path(args.chapters_json).expanduser().resolve()
                if not chapters_json.is_file():
                    log.error("[TEX] chapters_json not found: %s", chapters_json)
                    return 2
            else:
                cand = workdir_p / "themes" / "chapters_llm.json"
                if cand.is_file():
                    chapters_json = cand

            variants_root = workdir_p / "variants"
            if not variants_root.is_dir():
                log.error("[TEX] variants/ directory not found in workdir: %s", variants_root)
                variants_root = workdir_p / "variants"
                variants_root.mkdir(parents=True, exist_ok=True)

                neutral_dir = variants_root / "neutral"
                neutral_dir.mkdir(parents=True, exist_ok=True)

                target = neutral_dir / "VariantsGenerated_neutral.csv"
                if not target.is_file():
                    candidates = [
                        workdir_p / "neutral" / "NeutralGenerated.csv",
                        workdir_p / "neutral" / "NeutralGenerated_neutral.csv",
                    ]
                    for p in candidates:
                        if p.is_file():
                            target.write_bytes(p.read_bytes())
                            break
                    if not target.is_file():
                        for p in workdir_p.rglob("*.csv"):
                            if "neutral" in p.name.lower() and p.is_file():
                                target.write_bytes(p.read_bytes())
                                break
                
            variant_types = ["neutral"] + sorted(
                p.name for p in variants_root.iterdir()
                if p.is_dir() and p.name != "neutral"
            )

            for vt in variant_types:
                variant_dir = variants_root / vt
                csv_path = variant_dir / f"VariantsGenerated_{vt}.csv"
                if vt != "neutral":
                    if not csv_path.is_file():
                        log.warning("[TEX] skip %s: missing %s", vt, csv_path.name)
                        continue
                    dfv = pd.read_csv(csv_path, sep=";")
                    if dfv.empty or "Generated_Text" not in dfv.columns:
                        log.warning("[TEX] skip %s: invalid CSV %s", vt, csv_path.name)
                        continue
                    if dfv["Generated_Text"].astype(str).str.strip().eq("").any():
                        log.warning("[TEX] skip %s: incomplete Generated_Text in %s", vt, csv_path.name)
                        continue

                build_merged_tex_from_csv(
                    workdir=workdir_p,
                    variant_name=vt,
                    document_title=str(args.topic) if args.topic else workdir_p.name,
                    author="",
                    chapters_json=chapters_json,
                    themes_json=None,
                    add_orphans_chapter=True,
                    show_entry_numbers=not bool(args.no_entry_numbers),
                    show_snde=not bool(args.no_snde),
                    verbose=bool(args.verbose),
                )

        if not args.quiet:
            print("[DONE]")

        return 0
    finally:
        stack.close()

if __name__ == "__main__":
    raise SystemExit(main())
