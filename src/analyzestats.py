"""
narremgen.analyzestats
=====================
Statistical analysis utilities for generated narratives and code distributions.

Offers compact metrics for SN/DE diversity, lexical richness, sentiment and
emotion profiles, and length statistics, with optional per-text exports and
summary tables for neutral and variant corpora.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import slugify_topic

import logging
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.pipelines").setLevel(logging.ERROR)
logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)
logger_ = logger.info


_EMO_TOOLS = None
_MPL_IMPORTED = False
_LEXICALRICHNESS = None


def _get_lexicalrichness(verbose: bool = False):
    global _LEXICALRICHNESS
    if _LEXICALRICHNESS is not None:
        return _LEXICALRICHNESS
    try:
        from lexicalrichness import LexicalRichness
    except Exception as e:
        if verbose:
            logger.warning("LexicalRichness disabled (missing dependency): %s", e)
        _LEXICALRICHNESS = None
        return None
    _LEXICALRICHNESS = LexicalRichness
    return _LEXICALRICHNESS

def _get_emotion_tools(verbose=False):
    """Lazy import heavy emotion models."""
    global _EMO_TOOLS
    if _EMO_TOOLS is not None:
        return _EMO_TOOLS
    try:
        from transformers import pipeline as hf_pipeline
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        if verbose:
            logger_("!! Emotion stats disabled (missing heavy packages)")
        _EMO_TOOLS = None
        return None

    _EMO_TOOLS = {
        "hartmann": hf_pipeline("text-classification",
                                model="j-hartmann/emotion-english-distilroberta-base",
                                top_k=None),
        "go": hf_pipeline("text-classification",
                          model="SamLowe/roberta-base-go_emotions",
                          top_k=None),
        "vader": SentimentIntensityAnalyzer(),
    }
    return _EMO_TOOLS

def _get_plot_tools(verbose=False):
    """Lazy import matplotlib only if plots are requested."""
    global _MPL_IMPORTED
    if _MPL_IMPORTED:
        return True
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        _MPL_IMPORTED = True
        return True
    except Exception as e:
        if verbose:
            logger_("!! Plotting disabled (matplotlib missing)")
        return False


EMOTION_CATS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust']

def _compute_diversity_index(series: pd.Series, verbose: bool = False):
    """
    Compute a diversity index and Herfindahl concentration score for categorical data.

    This internal utility function takes a pandas Series representing categorical labels
    (e.g., SN or DE codes) and computes:
    1. The percentage of unique categories used.
    2. The Herfindahl concentration index, defined as the sum of squared frequency shares.

    Parameters
    ----------
    series : pandas.Series
        Series containing categorical identifiers (e.g., `Code_SN` or `Code_DE`).
    verbose : bool, optional
        If True, prints a short summary of the diversity and concentration results.

    Returns
    -------
    tuple[float, float]
        A tuple `(diversity_pct, concentration_index)` where:
        - `diversity_pct`: the ratio (in %) of unique categories to total items,
        - `concentration_index`: Herfindahl index measuring concentration of use.
    """
    counts = series.value_counts(normalize=True)
    diversity_pct = len(counts) / len(series) * 100 if len(series) else 0
    herfindahl = (counts ** 2).sum()
    if verbose: logger_(f"Diversity: {diversity_pct:.1f}%  |  Herfindahl Index: {herfindahl:.4f}")
    return diversity_pct, herfindahl


def _plot_distribution(counts: pd.Series, title: str, output_path: str | None = None,
                       show_plot: bool = False, save_plot: bool = True,
                       verbose=False):
    """
    Plot and optionally save a bar chart of categorical frequency distributions.

    Parameters
    ----------
    counts : pandas.Series
        A Series with category labels as index and their corresponding frequencies as values.
    title : str
        Title of the plot, typically including the topic or code type (e.g., "SN Distribution").
    output_path : str | None, optional
        Path to save the PNG image. If None, the plot is not saved.
    show_plot : bool, optional
        If True, displays the plot interactively (default False).
    save_plot : bool, optional
        If True, saves the plot to disk as a PNG image (default True).

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object.
    """
    if not _get_plot_tools(verbose=verbose): return
    import matplotlib.pyplot as plt
    if counts is None or counts.empty:
        if verbose: logger_(f"[warn] Empty dataframe passed to _plot_distribution for {title}. Skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title(title)
    ax.set_xlabel("Code")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_plot and output_path:
        plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)
    return fig


def _export_distribution_csv(counts: pd.Series, ref_df: pd.DataFrame, output_path: str, code_type: str):
    """
    Export a frequency distribution of codes to a CSV summary file.
    """

    counts = counts.reset_index()
    counts.columns = [f"Code_{code_type}", "Count"]

    df = pd.DataFrame({
        f"Code_{code_type}": counts[f"Code_{code_type}"],
        "Count": counts["Count"],
        "Percentage": (counts["Count"] / counts["Count"].sum() * 100).round(2)
    })

    if "Name" in ref_df.columns:
        df = df.merge(ref_df[["Code", "Name"]],
                      left_on=f"Code_{code_type}", right_on="Code", how="left")
        df = df.drop(columns=["Code"]).rename(columns={"Name": f"Name_{code_type}"})

    df.to_csv(output_path, index=False)
    return output_path


def _format_percentage(value: float | int, scale: bool = True) -> str:
    """
    Format numeric values as percentage strings with consistent precision.

    Parameters
    ----------
    value : float | int
        The numeric value to format as a percentage (e.g., 0.346 or 34.6).
    scale : bool, optional
        If True, the input is assumed to be a ratio (0-1) and multiplied by 100
        before formatting (default True).

    Returns
    -------
    str
        The formatted percentage string, always including one decimal place.
    """
    try:
        if scale:
            value = value * 100
        return f"{float(value):.1f}%"
    except Exception:
        return "0.0%"


def analyze_sn_de_distribution(mapping_file, sn_ref_file, de_ref_file,
                               output_dir="outputs", topic=None,
                               save_plot=True, show_plot=False, verbose=False):
    """
    Compute and visualize the distribution of SN (narrative structures) and DE (emotional dynamics) codes.

    This function loads the Mapping CSV file and merges it with the official SN and DE
    reference tables to obtain descriptive names and categories. It computes the relative
    frequency of each code, exports summary tables as CSV, and creates optional bar plots
    to visualize code usage across the dataset. It also calculates basic diversity metrics
    such as percentage of unique codes and Herfindahl concentration indices.

    Parameters
    ----------
    mapping_file : str
        Path to the Mapping CSV containing at least columns `Code_SN` and `Code_DE`.
    sn_ref_file : str
        Path to the official SN reference CSV (must include columns `Code` and `Name`).
    de_ref_file : str
        Path to the official DE reference CSV (must include columns `Code` and `Name`).
    output_dir : str, optional
        Directory where output CSVs and plots will be stored (default `'outputs'`).
    topic : str | None, optional
        Optional tag appended to output filenames and plot titles.
    save_plot : bool, optional
        If True, saves PNG bar plots for SN and DE distributions (default True).
    show_plot : bool, optional
        If True, displays plots interactively instead of just saving them (default False).
    verbose : bool, optional
        If True, prints code distribution summaries and diversity statistics.

    Returns
    -------
    dict
        Dictionary containing:
        {
            'SN_diversity_pct': float,
            'DE_diversity_pct': float,
            'SN_concentration_index': float,
            'DE_concentration_index': float,
            'sn_file': str,  
            'de_file': str   
        }

    Notes
    -----
    - The diversity percentage indicates how many unique codes are used relative to total possibilities.
    - The Herfindahl index measures concentration: higher values mean less diversity.
    - Generated CSVs can be reused for corpus-level comparisons or cross-topic studies.
    """
    df = pd.read_csv(mapping_file, sep=";")
    sn_ref = pd.read_csv(sn_ref_file, sep=";")
    de_ref = pd.read_csv(de_ref_file, sep=";")

    counts_sn = df["Code_SN"].value_counts()
    counts_de = df["Code_DE"].value_counts()

    if counts_sn is None or counts_sn.empty or counts_de is None or counts_de.empty:
        if verbose: logger_("[warn] Empty SN/DE counts detected. Skipping distribution plots.")
        return
    
    div_sn, conc_sn = _compute_diversity_index(df["Code_SN"], verbose=verbose)
    div_de, conc_de = _compute_diversity_index(df["Code_DE"], verbose=verbose)

    topic_slug = slugify_topic(topic)
    topic_tag = f"_{topic_slug}" if topic else ""
    sn_csv = f"{output_dir}/SN_Distribution{topic_tag}.csv"
    de_csv = f"{output_dir}/DE_Distribution{topic_tag}.csv"
    sn_plot = f"{output_dir}/SN_Distribution{topic_tag}.png"
    de_plot = f"{output_dir}/DE_Distribution{topic_tag}.png"

    _export_distribution_csv(counts_sn, sn_ref, sn_csv, "SN")
    _export_distribution_csv(counts_de, de_ref, de_csv, "DE")

    _plot_distribution(counts_sn, f"SN Distribution {topic_slug or ''}", sn_plot,
                       show_plot=show_plot, save_plot=save_plot)
    _plot_distribution(counts_de, f"DE Distribution {topic_slug or ''}", de_plot,
                       show_plot=show_plot, save_plot=save_plot)

    if verbose:
        logger_(f"SN diversity: {_format_percentage(div_sn)} "
              f"| DE diversity: {_format_percentage(div_de)}")

    return {
        "SN_diversity_pct": div_sn,
        "DE_diversity_pct": div_de,
        "SN_concentration_index": conc_sn,
        "DE_concentration_index": conc_de,
        "sn_file": sn_csv,
        "de_file": de_csv,
    }


def _basic_tokenize(text: str) -> list[str]:
    """
    Very simple whitespace tokenizer for fallback lexical statistics.
    """
    return str(text).split()


def analyze_text_outputs(
    source: str | Path | pd.DataFrame,
    text_col: str = "Text",
    label: str | None = None,
    use_lexicalrichness: bool = True,
    use_emotions: bool = False,
    do_plots: bool = False,
    sep: str = ";",
    verbose: bool = False,
    per_text_out: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute textual statistics for a corpus (neutral or variant).

    Metrics include:
    - Length measures: tokens, characters, sentences, words.
    - Lexical richness (TTR, RTTR, CTTR, Maas, HDD, MTLD).
    - Optionally: emotion analysis (Hartmann, GoEmotions) and VADER sentiment.

    Parameters
    ----------
    source : str | Path | DataFrame
        CSV path or DataFrame containing the texts.
    text_col : str
        Column with raw text content.
    label : str or None
        Corpus identifier for summaries.
    use_lexicalrichness : bool
        Enable advanced lexical metrics.
    use_emotions : bool
        Enable emotion / sentiment metrics.
    sep : str
        CSV separator for loading `source`.
    verbose : bool
        Debug logging.
    per_text_out : str or Path or None
        If provided, save all per-text metrics.

    Returns
    -------
    pandas.DataFrame
        One-row summary of corpus-level statistics.
    """
  
    emo_hartmann = emo_go = vader = None
    if use_emotions:
        tools = _get_emotion_tools(verbose=verbose)
        if tools is None:
            use_emotions = False
        else:
            emo_hartmann = tools.get("hartmann")
            emo_go = tools.get("go")
            vader = tools.get("vader")

    if do_plots:
        _get_plot_tools(verbose=verbose)

    LexicalRichness = None
    lr_available = bool(use_lexicalrichness)
    if lr_available:
        LexicalRichness = _get_lexicalrichness(verbose=verbose)
        if LexicalRichness is None:
            lr_available = False
            if verbose:
                logger_("[warn] lexicalrichness not available; using fallback lexical metrics.")

    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, sep=sep)
    else:
        df = source.copy()

    if text_col not in df.columns:
        if "Generated_Text" in df.columns:
            if verbose:
                logger_(
                    f"[info] text_col={text_col!r} no found, "
                    "using 'Generated_Text' instead."
                )
            text_col = "Generated_Text"
        else:
            raise ValueError(
                f"Column {text_col!r} not found in dataframe "
                f"and no obvious alternative is available."
            )

    series = df[text_col].fillna("").astype(str)
    series = series[series.str.strip() != ""]
    if series.empty:
        raise ValueError("No non-empty texts found for statistics.")

    def _word_count(text: str) -> int:
        return len(text.split())

    def _sent_count(text: str) -> int:
        spans = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return len(spans) if spans else 1

    rows: list[dict[str, float | int | str | None]] = []

    for idx, txt in series.items():
        txt = str(txt)

        tokens = _basic_tokenize(txt)
        n_tokens = len(tokens)
        n_chars = len(txt)
        n_words = _word_count(txt)
        n_sents = _sent_count(txt)

        rec: dict[str, float | int | str | None] = {}

        if "Num" in df.columns:
            try:
                rec["Num"] = int(df.loc[idx, "Num"])
            except Exception:
                rec["Num"] = None

        rec.update(
            {
                "len_tokens": float(n_tokens),
                "len_chars": float(n_chars),
                "word_count": float(n_words),
                "len_sentences": float(n_sents),
            }
        )

        if lr_available and LexicalRichness is not None and n_tokens >= 5:
            try:
                lex = LexicalRichness(txt)
                rec.update(
                    {
                        "TTR": float(lex.ttr),
                        "RTTR": float(lex.rttr),
                        "CTTR": float(lex.cttr),
                        "Maas": float(lex.Maas),
                        "HDD": float(lex.hdd(draws=42)),
                        "MTLD": float(lex.mtld(threshold=0.72)),
                    }
                )
            except Exception:
                lr_available = False
                if verbose:
                    logger_("[warn] lexicalrichness failed; switching to fallback metrics only.")

        if not lr_available:
            if n_tokens > 0:
                vocab = set(tokens)
                rec["TTR"] = len(vocab) / n_tokens
                counts = pd.Series(tokens).value_counts()
                hapax = (counts == 1).sum()
                rec["HapaxRatio"] = hapax / n_tokens
                if n_tokens > 1:
                    rec["HerdanC"] = np.log(len(vocab)) / np.log(n_tokens)
                else:
                    rec["HerdanC"] = np.nan
            else:
                rec["TTR"] = np.nan
                rec["HapaxRatio"] = np.nan
                rec["HerdanC"] = np.nan

            rec.setdefault("RTTR", np.nan)
            rec.setdefault("CTTR", np.nan)
            rec.setdefault("Maas", np.nan)
            rec.setdefault("HDD", np.nan)
            rec.setdefault("MTLD", np.nan)

        if use_emotions and emo_hartmann is not None and emo_go is not None:
            out7 = emo_hartmann(txt, truncation=True)[0]
            for d in out7:
                lab = d["label"]
                rec[f"emo7_{lab}"] = float(d["score"])

            out_go = emo_go(txt, truncation=True)[0]
            for d in out_go:
                lab = d["label"]
                rec[f"goemo_{lab}"] = float(d["score"])

            if vader is not None:
                vs = vader.polarity_scores(txt)
                for k, v in vs.items():
                    rec[f"vader_{k}"] = float(v)

        rows.append(rec)

    df_metrics = pd.DataFrame(rows)

    if label is not None:
        df_metrics.insert(0, "corpus", str(label))

    if per_text_out is not None:
        per_text_out = Path(per_text_out)
        per_text_out.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(per_text_out, sep=";", index=False, encoding="utf-8")
        if verbose:
            logger_(f"[STATS] Per-text metrics saved to {per_text_out}")

    summary: dict[str, float | int | str] = {}
    summary["n_texts"] = int(len(df_metrics))

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        vals = (
            pd.to_numeric(df_metrics[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        summary[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
        summary[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
        summary[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan

    for col in ("len_tokens", "len_chars", "word_count", "len_sentences"):
        if col in df_metrics.columns:
            vals = (
                pd.to_numeric(df_metrics[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            summary[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
            summary[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan

    if label is not None:
        summary["corpus"] = str(label)

    return pd.DataFrame([summary])


def compute_length_stats_for_corpus(
    df: pd.DataFrame,
    text_col: str,
    label: str,
) -> pd.DataFrame:
    """
    Compute basic length statistics for a corpus.

    Includes:
    - number of texts
    - token count (mean/median/std)
    - character count (mean/median/std)

    Parameters
    ----------
    df : pandas.DataFrame
        Containing text data.
    text_col : str
        Column to analyze.
    label : str
        Corpus identifier.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame summarizing corpus length stats.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column {text_col!r} missing in DataFrame for {label!r}")

    s = df[text_col].fillna("").astype(str)
    s = s[s.str.strip() != ""]

    if s.empty:
        return pd.DataFrame([{"corpus": label, "n_texts": 0}])

    lens_tokens = s.str.split().map(len)
    lens_chars = s.map(len)

    stats = {
        "corpus": label,
        "n_texts": int(len(lens_tokens)),
        "len_tokens_mean": float(lens_tokens.mean()),
        "len_tokens_median": float(lens_tokens.median()),
        "len_tokens_std": float(lens_tokens.std(ddof=1)),
        "len_chars_mean": float(lens_chars.mean()),
        "len_chars_median": float(lens_chars.median()),
        "len_chars_std": float(lens_chars.std(ddof=1)),
    }
    return pd.DataFrame([stats])


PACKAGE_METRICS = {
    "TTR",
    "RTTR",
    "CTTR",
    "HDD",
    "MTLD",
    "Maas",
    "HapaxRatio",
    "HerdanC",
}

DIRECT_METRICS = {
    "len_chars",
    "len_tokens",
    "len_sentences",
    "word_count",
}

def build_global_stats_table(
    stats_dict: dict[str, pd.DataFrame],
    list_variants: list[str],
    metric_source: str = "both",  # "both" | "package" | "direct"
    verbose: bool = True,
) -> pd.DataFrame | None:
    """
    Build a comparison table of corpus statistics across variants.

    This function takes a dict of variant-name -> metrics DataFrame and produces a
    single table aligned by metric name. It can optionally filter to package metrics
    (lexicalrichness) or to direct metrics.

    Parameters
    ----------
    stats_dict : dict[str, pandas.DataFrame]
        One DataFrame per variant.
    list_variants : list[str]
        Variants expected in the comparison; missing variants are skipped.
    metric_source : {"both", "package", "direct"}
        Filter which metric sources are included.
    verbose : bool, default False
        If True, print debug information.

    Returns
    -------
    pandas.DataFrame or None
        Comparison table indexed by metric, or None if no common metrics exist.

    Notes
    -----
    The output table is shaped like:

        metric | neutral | simple | formal | poetic
    """

    if not stats_dict:
        if verbose:
            logger_("[WARN] Empty stats_dict — skipping global stats.")
        return None

    present_variants = [vt for vt in list_variants if vt in stats_dict]
    if not present_variants:
        if verbose:
            logger_("[WARN] No variants found in stats_dict — skipping global stats.")
        return None

    common_mean_cols: set[str] | None = None
    for vt, df in stats_dict.items():
        mean_cols = {c for c in df.columns if c.endswith("_mean")}
        if common_mean_cols is None:
            common_mean_cols = mean_cols
        else:
            missing_here = common_mean_cols - mean_cols
            if missing_here and verbose:
                logger_(f"[WARN] {vt} missing metrics: {sorted(missing_here)}")
            common_mean_cols &= mean_cols

    if not common_mean_cols:
        if verbose:
            logger_("[WARN] No common *_mean metrics across stats files — aborting global table.")
        return None

    root_metrics = sorted(col.rsplit("_", 1)[0] for col in common_mean_cols)

    if metric_source == "package":
        root_metrics = [m for m in root_metrics if m in PACKAGE_METRICS]
    elif metric_source == "direct":
        root_metrics = [m for m in root_metrics if m in DIRECT_METRICS]
    elif metric_source == "both":
        pass
    else:
        raise ValueError(f"Unknown metric_source={metric_source!r} (expected 'both', 'package', 'direct').")

    if not root_metrics:
        if verbose:
            logger_(f"[WARN] No metrics left after filtering for metric_source={metric_source!r}.")
        return None

    if verbose:
        logger_(f"\n[DEBUG] ROOT METRICS FOUND ({metric_source}):")
        logger_(root_metrics)

    final_rows: list[dict[str, str]] = []

    first_df = stats_dict[present_variants[0]]

    for metric in root_metrics:
        mean_col = metric + "_mean"
        std_col = metric + "_std"

        if std_col not in first_df.columns:
            if verbose:
                logger_(f"[WARN] Missing std column for metric {metric} — skipping.")
            continue

        row: dict[str, str] = {"metric": metric}

        neu_sel = first_df.loc[first_df["corpus"] == "neutral", [mean_col, std_col]]
        if neu_sel.empty:
            if verbose:
                logger_(f"[WARN] No neutral row for metric {metric} — skipping.")
            continue
        mean_neu = neu_sel[mean_col].iloc[0]
        std_neu = neu_sel[std_col].iloc[0]
        row["neutral"] = f"{mean_neu:.2f} ({std_neu:.2f})"

        for vt in present_variants:
            df = stats_dict[vt]
            if mean_col not in df.columns or std_col not in df.columns:
                if verbose:
                    logger_(f"[WARN] {vt} missing {mean_col} or {std_col} — skipping for this metric.")
                continue

            sel = df.loc[df["corpus"] == vt, [mean_col, std_col]]
            if sel.empty:
                if verbose:
                    logger_(f"[WARN] {vt}: no row for metric {metric} — skipping for this variant.")
                continue

            mean_val = sel[mean_col].iloc[0]
            std_val = sel[std_col].iloc[0]
            row[vt] = f"{mean_val:.2f} ({std_val:.2f})"

        final_rows.append(row)

    if not final_rows:
        if verbose:
            logger_("[WARN] No rows built for global stats — returning None.")
        return None

    global_stats = pd.DataFrame(final_rows).set_index("metric")

    ordered_cols = ["neutral"] + present_variants
    global_stats = global_stats[[c for c in ordered_cols if c in global_stats.columns]]

    return global_stats
