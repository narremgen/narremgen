"""
narremgen.narratives
===================
Batch and full-corpus narrative generation from SN/DE mappings and context.

Implements giga-prompt construction, batched LLM calls, and flexible merging
of batch files into final documents (text, Markdown, LaTeX, DOCX, PDF),
controlling header style and extra instructions for the Narremgen stories.
"""

import glob
import os, re
import pandas as pd
from typing import List
from .llmcore import LLMConnect
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

def generate_narratives_batch(
    mapping_file: str,
    sn_file: str,
    de_file: str,
    advice_file: str,
    context_file: str,
    style_file: str = "style.txt",
    plain_text: bool = True,
    examples_file: str = "examples.txt",
    start_line: int = 0,
    batch_size: int = 5,
    language: str = "en",
    max_tokens: int | None = 8000,
    output_dir: str = "./outputs/",
    target_words: int = 400,
    dialogue_mode: str = "single",
    extra_instructions: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Generate a batch of narrative texts using aligned Mapping, Advice, Context,
    SN and DE data.

    The function constructs a single “giga-prompt” that includes:
    - global style instructions,
    - optional extra instructions,
    - few-shot examples,
    - the full SN/DE reference tables,
    - a slice of rows from Mapping/Advice/Context (Num, codes, topic, advice, sentence, context).

    This giga-prompt is sent to ``LLMConnect.get_global().safe_chat_completion``.
    The resulting text (raw, unparsed) is written to a numbered batch file:
    ``outputs/batches_text/batch_<topic>_<start>_<end>.txt``.

    Parameters
    ----------
    mapping_file : str
        Path to the Mapping CSV (must include Num, Code_SN, Code_DE).
    sn_file : str
        Path to the SN reference CSV (Code, Name, Narrative_Structure).
    de_file : str
        Path to the DE reference CSV (Code, Name, Emotional_Sequence).
    advice_file : str
        Path to the Advice CSV (Num, Topic, Advice, Sentence).
    context_file : str
        Path to the Context CSV (Num, Character, Location, etc.).
    style_file : str
        Path to a text file containing global style and safety rules.
    plain_text : bool
        If True, request a single-paragraph narrative (no visible markup).
    examples_file : str
        Few-shot narrative examples.
    start_line : int
        Starting index (0-based) in the Mapping CSV.
    batch_size : int
        Number of rows to process in this batch.
    language : str
        Output language requested from the model.
    max_tokens : int or None
        Maximum completion tokens for the model.
    output_dir : str
        Root directory where batch files will be written.
    target_words : int
        Target length of the generated narrative (soft constraint).
    dialogue_mode: str ("none" | "single" | "short")
        Controls the presence and shape of direct dialogue in neutral texts.
    extra_instructions : str or None
        Optional additional instructions appended to the style block.
    verbose : bool
        If True, detailed debug logs are printed.

    Returns
    -------
    str
        Path to the generated batch file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mapping = pd.read_csv(mapping_file, sep=";")
    sn = pd.read_csv(sn_file, sep=";").set_index("Code")
    de = pd.read_csv(de_file, sep=";").set_index("Code")
    context = pd.read_csv(context_file, sep=";").set_index("Num")
    advice = pd.read_csv(advice_file, sep=";").set_index("Num")
    
    style_text = open(style_file, encoding="utf-8").read()

    style_text += """

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
  there is no need to describe equipment except if required like with bicycle.
- When a topic sounds metaphorical (for example “jumping into the void”), 
  treat it as a figure of speech or as a fully controlled sport activity, and 
  emphasise realistic, safe behaviour in real life. 
[/SAFETY_RULES]
"""
   
    examples_text = open(examples_file, encoding="utf-8").read()
    if plain_text:
        style_text += (
            f"\n\nIMPORTANT: Utilise le balisage pour guider la création du contenu, "
            "mais NE PAS afficher les balises, commentaires, ni explications. "
            "Produit uniquement le texte brut, sous forme d’un paragraphe unique par récit "
            f"({target_words-50}-{target_words+50} mots)."
        )
    else:
        style_text += (
            f"\n\nIMPORTANT: Chaque section doit respecter environ {target_words//5} mots, "
            f"pour un total de {target_words-50}-{target_words+50} mots par récit."
        )

    if extra_instructions and extra_instructions.strip():
        style_text += "\n\n" + extra_instructions

    if dialogue_mode == "none":
        style_text += """
\n[DIALOGUE_RULES]
- Do NOT use any direct speech or quotation marks.
- Narration only: no dialogue lines, no quoted sentences.
[/DIALOGUE_RULES]
"""
    elif dialogue_mode == "single":
        style_text += """
\n[DIALOGUE_RULES]
- Include exactly ONE line of direct speech (one sentence between quotes).
- This sentence must convey or receive the main advice.
- No other dialogue lines are allowed.
[/DIALOGUE_RULES]
"""
    elif dialogue_mode == "short":
        style_text += """
\n[DIALOGUE_RULES]
- Include a very short dialogue of 2–4 lines of direct speech.
- The dialogue must revolve around giving, discussing, or applying the advice.
- Keep the rest of the narration concise and focused.
[/DIALOGUE_RULES]
"""

    sn_ref = pd.read_csv(sn_file, sep=";")
    de_ref = pd.read_csv(de_file, sep=";")

    style_text += "\n\n--- OFFICIAL SN CODES ---\n"
    style_text += sn_ref.to_csv(sep=";", index=False)

    style_text += "\n\n--- OFFICIAL DE CODES ---\n"
    style_text += de_ref.to_csv(sep=";", index=False)

    batch = mapping.iloc[start_line:start_line + batch_size]

    prompt_parts: List[str] = []
    prompt_parts.append(style_text)
    prompt_parts.append("\n--- EXAMPLES ---\n")
    prompt_parts.append(examples_text)
    prompt_parts.append(
        f"\nNow generate narratives in {language.upper()} for the following rows.\n"
        "!! Each narrative MUST strictly apply the provided SN (narrative structure) "
        "and DE (emotional sequence) for its line.\n"
        "!! Do not invent new structures; follow the definitions above.\n"
    )

    for _, row in batch.iterrows():
        num = row["Num"]
        sn_code, de_code = row["Code_SN"], row["Code_DE"]

        if sn_code not in sn.index:
            if verbose:
                logger_(
                    f"[WARN] SN code {sn_code!r} not found in {sn_file}, "
                    "falling back to 'SN1'."
                )
            sn_code = "SN1"

        if de_code not in de.index:
            if verbose:
                logger_(
                    f"[WARN] DE code {de_code!r} not found in {de_file}, "
                    "falling back to 'DE1'."
                )
            de_code = "DE1"

        sn_name = sn.loc[sn_code, "Name"]
        sn_struct = sn.loc[sn_code, "Narrative_Structure"]
        de_name = de.loc[de_code, "Name"]
        de_struct = de.loc[de_code, "Emotional_Sequence"]

        adv = advice.loc[num]
        ctx = context.loc[num]

        line_block = f"""
                    --- LINE {num} ---
                    Number: {num}
                    Topic: {adv['Topic']}
                    Advice: {adv['Advice']}
                    Sentence: "{adv['Sentence']}"
                    SN {sn_code} - {sn_name} ({sn_struct})
                    DE {de_code} - {de_name} ({de_struct})
                    Context: {ctx['Character']}, {ctx['Presence']}, {ctx['Location']}, {ctx['Sensation']}, {ctx['Time']}, {ctx['Moment']}, {ctx['First_Name']}
                    """
        prompt_parts.append(line_block)

        giga_prompt = "\n".join(prompt_parts)

    prompt_tokens = LLMConnect.estimate_tokens(giga_prompt)

    expected_output_tokens = target_words # (1 word ~ 1 token = approx warning)
    total_estimate = prompt_tokens + expected_output_tokens * batch_size
    max_model_tokens = max_tokens or 8000  # fallback

    if total_estimate > max_model_tokens * 0.8:
        model = LLMConnect.get_model("NARRATIVE")
        if verbose:
            usage_pct = round(total_estimate / max_model_tokens * 100, 1)
            logger_(f"!! Warning: batch_size={batch_size} risk to saturate {model}. "
                  f"({usage_pct}% de la limite)")
            logger_(f"   Estimation : {total_estimate} tokens "
                  f"(prompt ~{prompt_tokens}, output ~{expected_output_tokens * batch_size})")
            logger_(f"   Limit model : {max_model_tokens} tokens")


    topic_slug = os.path.splitext(os.path.basename(advice_file))[0].replace("Advice_", "")

    max_retries = 3
    result = None
    for attempt in range(1, max_retries + 1):
        result = LLMConnect.get_global().safe_chat_completion(            
            model=LLMConnect.get_model("NARRATIVE"),
            messages=[{"role": "user", "content": giga_prompt}],
            max_tokens=max_tokens
        )

        if result is not None:
            if verbose:
                logger_(f"Generation succeeded at attempt {attempt} "
                    f"for lines {start_line+1}-{start_line+batch_size}")
            break
        else:
            if verbose:
                logger_(f"!! Attempt {attempt} failed for lines "
                    f"{start_line+1}-{start_line+batch_size}")

    if result is None:
        placeholder = "\n\n".join(
            f"[GENERATION_FAILED for line {i}]"
            for i in range(start_line+1, start_line+batch_size+1)
        )
        result = placeholder
        if verbose:
            logger_(f"!!! All {max_retries} attempts failed - writing placeholder text.")

    batch_dir = os.path.join(output_dir, "batches_text")
    os.makedirs(batch_dir, exist_ok=True)
    batch_file = os.path.join(
        batch_dir,
        f"batch_{topic_slug}_{start_line+1}_{start_line+batch_size}.txt"
    )
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write(result)

    if verbose:
        logger_(f"[narratives] Batch stored in {batch_dir}")

    if verbose:
        logger_(f"Narratives generated for lines {start_line+1} to {start_line+batch_size}")
        logger_(f"Output saved: {batch_file}")

    return batch_file 


def generate_narratives(
    mapping_file: str,
    sn_file: str,
    de_file: str,
    advice_file: str,
    context_file: str,
    style_file: str = "style.txt",
    examples_file: str = "examples.txt",
    plain_text: bool = True,
    start_line: int = 0,
    end_line: int = 10,
    batch_size: int = 5,
    language: str = "fr",
    max_tokens: int | None = 8000,
    output_dir: str = "./outputs/",
    final_output: str = "merged.txt",
    output_format: str = "txt",   # "txt", "md", "tex", "docx", "pdf", "append"
    header_style: str = "full",   # "full", "simple", "none"
    target_words: int = 400,
    dialogue_mode: str = "single",
    extra_instructions: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Generate narrative texts for a full dataset by splitting it into batches
    and merging all batch outputs into a final document.

    This function:
    1. Iterates from ``start_line`` to ``end_line`` in increments of ``batch_size``.
    2. Calls :func:`generate_narratives_batch` for each segment.
    3. Collects the batch text files.
    4. Produces a final merged output using :func:`merge_batches`, or appends
       to an existing file if ``output_format="append"``.

    Parameters
    ----------
    mapping_file : str
        Path to the Mapping CSV with SN/DE codes.
    sn_file : str
        Path to the SN reference CSV.
    de_file : str
        Path to the DE reference CSV.
    advice_file : str
        Path to the Advice CSV.
    context_file : str
        Path to the Context CSV.
    style_file : str
        Global style and safety instructions.
    examples_file : str
        Few-shot narrative examples.
    plain_text : bool
        If True, instruct the model to output a single raw paragraph.
    start_line : int
        Starting index (0-based) of the slice to generate.
    end_line : int
        End index (exclusive) for generation.
    batch_size : int
        Number of items per batch.
    language : str
        Output language passed to each batch.
    max_tokens : int or None
        Maximum completion tokens allowed for each batch generation.
    output_dir : str
        Directory where batch files and final merged files will be created.
    final_output : str
        Base filename of the merged result (extension depends on output_format).
    output_format : {"txt","md","tex","docx","pdf","append"}
        Format of the final merged document.
    header_style : {"full","simple","none"}
        Controls how headers in batch files are cleaned during merging.
    target_words : int
        Approximate narrative length for each item.
    dialogue_mode: str ("none" | "single" | "short")
        Controls the presence and shape of direct dialogue in neutral texts.
    extra_instructions : str or None
        Additional global (style, context) instructions injected in the prompt.
    verbose : bool
        If True, print detailed progress logs.

    Returns
    -------
    str
        Path to the final merged narrative file.
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    for s in range(start_line, end_line, batch_size):
        e = min(s + batch_size, end_line)
        file = generate_narratives_batch(
            mapping_file, sn_file, de_file, advice_file, context_file,
            style_file=style_file,
            examples_file=examples_file,
            plain_text=plain_text,
            start_line=s,
            batch_size=e - s,
            language=language,            
            max_tokens=max_tokens,
            output_dir=output_dir,
            target_words=target_words,
            dialogue_mode=dialogue_mode,
            extra_instructions=extra_instructions,
            verbose=verbose
        )
        batch_files.append(file)

    base_name, ext = os.path.splitext(final_output)
    if output_format == "append":
        if not ext:
            ext = ".txt"  
        final_path = os.path.join(output_dir, base_name + ext)
        with open(final_path, "a", encoding="utf-8") as outfile:
            for f in batch_files:
                text = open(f, encoding="utf-8").read()
                text = _filter_headers(text, header_style)
                outfile.write(text.strip() + "\n\n")
        if verbose:
            logger_(f"Final file updated by append : {final_path}")
        return final_path
    else:
        merged_filename = base_name if base_name else "merged_output"
        merged_path = merge_batches(
            output_dir=output_dir,
            merged_filename=merged_filename,   
            output_format=output_format,
            header_style=header_style
        )
        if verbose:
            logger_(f"Final file generated : {merged_path}")
        return merged_path

def merge_batches(output_dir="outputs",
                  merged_filename="merged_output",
                  output_format="txt",
                  header_style="full",
                  verbose=False) -> str:
    """
    Merge multiple batch output files into a single document in the chosen format.

    This function scans a directory for files matching the pattern `batch_*.txt`,
    concatenates them in numeric order, optionally cleans headers, and writes the result
    as a merged file. Supported output formats include plain text, Markdown, LaTeX, DOCX,
    and PDF. Non-text outputs require `python-docx` or `reportlab` to be installed.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing batch text files (default `'./outputs/'`).
    merged_filename : str, optional
        Output filename (without extension) for the merged document (default `'merged'`).
    output_format : {"txt", "md", "tex", "docx", "pdf"}, default "txt"
        Output format to produce.
    header_style : {"none", "strip", "normalize"}, default "strip"
        How to handle batch headers when concatenating.
    verbose : bool, default False
        If True, print progress information.

    Returns
    -------
    str
        Path to the produced merged file.

    Notes
    -----
    - The merging order is based on numeric sorting of filenames (for example: `batch_1`,
    `batch_2`, `batch_3`).
    - The function attempts to detect encoding and ensures UTF-8 output where applicable.
    """
   
    candidates = [
        os.path.join(output_dir, "batches_text"),
        output_dir,
    ]
    search_dir = next((d for d in candidates if os.path.isdir(d)), output_dir)

    patterns = [
        "batch_*.txt",
        "batch_Filtered*.txt",
        "batch_*Filtered*.txt",
    ]
    batch_files = []
    for pat in patterns:
        batch_files.extend(glob.glob(os.path.join(search_dir, pat)))
    batch_files = sorted(set(batch_files), key=_sort_key)

    if not batch_files:
        if verbose:
            logger_(f"!! No batch file found in {search_dir}")
        return os.path.join(search_dir, f"{merged_filename}.{output_format}")

    merged_path = os.path.join(output_dir, f"{merged_filename}.{output_format}")

    if output_format in ["txt", "md", "tex"]:
        with open(merged_path, "w", encoding="utf-8") as outfile:
            for path in batch_files:
                text = open(path, encoding="utf-8").read()
                text = _filter_headers(text, header_style)
                outfile.write(text.strip() + "\n\n")
        if verbose:
            logger_(f"Merged File stored : {merged_path}")
        return merged_path

    elif output_format == "docx":
        try:
            from docx import Document
        except ImportError:
            if verbose:
                logger_("!! Module 'python-docx' missing. Fallback in .txt")
            return merge_batches(output_dir, merged_filename, "txt", header_style)

        doc = Document()
        for path in batch_files:
            text = open(path, encoding="utf-8").read()
            text = _filter_headers(text, header_style)
            doc.add_paragraph(text.strip())
            doc.add_page_break()
        doc.save(merged_path)
        if verbose:
            logger_(f"Merged File DOCX stored : {merged_path}")
        return merged_path

    elif output_format == "pdf":
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            if verbose:
                logger_("!! Module 'reportlab' missing. Fallback in .txt")
            return merge_batches(output_dir, merged_filename, "txt", header_style)

        doc = SimpleDocTemplate(merged_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        for path in batch_files:
            text = open(path, encoding="utf-8").read()
            text = _filter_headers(text, header_style)
            story.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))
            story.append(Spacer(1, 20))
        doc.build(story)
        if verbose:
            logger_(f"Merged file PDF stored : {merged_path}")
        return merged_path

    else:
        if verbose:
            logger_(f"!! Unsupported Format : {output_format}. Fallback in .txt")
        return merge_batches(output_dir, merged_filename, "txt", header_style)


def _filter_headers(text: str, header_style: str) -> str:
    """
    Clean or retain headers and comment lines from a generated text block.

    This internal helper is used during merging to adjust the visibility of batch headers,
    comments, or section markers. It removes or simplifies formatting elements based on
    the selected header style, making the final document uniform and readable.

    Parameters
    ----------
    text : str
        The text content to process (may include headers, comments, or LaTeX/Markdown markers).
    header_style : {'full','simple','none'}
        Controls how much of the header structure is retained:
        - `'full'`: keep all headers and comments.
        - `'simple'`: keep light section headers only (Markdown ### or LaTeX \\subsubsection*),
        remove comments and metadata lines.
        - `'none'`: remove all headers and comments entirely.

    Returns
    -------
    str
        The cleaned text block formatted according to the selected header style.

    Notes
    -----
    - This function is mainly used by `merge_batches()` before assembling the final document.
    - It is not intended for public use and assumes well-structured narrative output files.
    """

    if header_style == "simple":
        return "\n".join(
            line for line in text.splitlines()
            if line.startswith("\\subsubsection*") or line.startswith("###") or not line.startswith("%")
        )
    elif header_style == "none":
        return "\n".join(
            line for line in text.splitlines()
            if not line.startswith("\\") and not line.startswith("%") and not line.startswith("###")
        )
    return text


def _sort_key(filename):
    """
    Extract a numeric sort key from a batch filename.

    Parameters
    ----------
    filename : str
        Batch filename (expected format like `batch_12.txt` or `batch_12_extra.txt`).

    Returns
    -------
    int
        The numeric index if found, otherwise a large sentinel value to push the file last.
    """

    
    m = re.search(r"_(\d+)_", filename)
    return int(m.group(1)) if m else 0
