"""
narremgen.data
=============
LLM-driven generation of the core Advice, Context, and Mapping CSV tables.

Wraps LLMConnect calls and post-processing to generate clean,
semicolon-separated datasets with stable naming conventions, forming the
entry point of the Narremgen pipeline.
"""

import os, re, unicodedata
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from .llmcore import LLMConnect
from .utils import save_output
from .utils import slugify_topic
from .utils import postprocess_csv_text_basic

import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

def generate_advice_titles_plan(topic: str,
                               n_total: int,
                               advice_context: str | None = None,
                               verbose: bool = False,
                               sim_ratio_threshold: float = 0.90,
                               max_rounds: int = 40) -> list[str]:
    """Generate a deduplicated list of advice titles ahead of CSV generation."""

    def _ascii_norm(s: str) -> str:
        t = unicodedata.normalize("NFKD", s or "")
        t = t.encode("ascii", "ignore").decode("ascii", errors="ignore")
        t = t.casefold()
        t = re.sub(r"[^a-z0-9\s]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _dedup_titles(titles: list[str]) -> list[str]:
        kept: list[str] = []
        kept_norm: list[str] = []
        for raw in titles:
            t = " ".join((raw or "").strip().split())
            if not t:
                continue
            n = _ascii_norm(t)
            if not n:
                continue
            dup = False
            for kn in kept_norm:
                if n == kn or SequenceMatcher(None, n, kn).ratio() >= sim_ratio_threshold:
                    dup = True
                    break
            if dup:
                continue
            kept.append(t.replace(";", " ").strip())
            kept_norm.append(n)
        return kept

    def _extract_seed(ctx: str | None) -> list[str]:
        if not ctx:
            return []
        m = re.search(r"<ADVICE_PLAN>(.*?)</ADVICE_PLAN>", ctx, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        out: list[str] = []
        for ln in m.group(1).splitlines():
            t = re.sub(r"^\s*(?:[-*]|\d+[.)-])\s*", "", ln).strip()
            t = " ".join(t.split())
            if t:
                out.append(t)
        return out

    titles = _dedup_titles(_extract_seed(advice_context))

    llm = LLMConnect.get_global()
    model = LLMConnect.get_model("ADVICE")
    ctx = (advice_context or "").strip()
    ctx_block = f"<ADVICE_CONTEXT>\n{ctx}\n</ADVICE_CONTEXT>\n\n" if ctx else ""

    rounds = 0
    while len(titles) < n_total and rounds < max_rounds:
        rounds += 1
        remaining = n_total - len(titles)
        n_request = min(120, max(20, remaining + 10))
        avoid = "\n".join(f"- {t}" for t in titles[-80:])

        prompt = f"""
Generate {n_request} distinct advice TITLES for the topic:
{topic}

{ctx_block}Do not repeat or paraphrase any title below:
{avoid}

Rules:
- One title per line
- No numbering, no bullets, no markdown, no extra text
- Do not use ';' inside titles
""".strip()

        text = llm.safe_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        ) or ""

        new_titles: list[str] = []
        for ln in text.splitlines():
            t = re.sub(r"^\s*(?:[-*]|\d+[.)-])\s*", "", ln).strip()
            t = " ".join(t.split()).replace(";", " ").strip()
            if t:
                new_titles.append(t)

        prev = len(titles)
        titles = _dedup_titles(titles + new_titles)

        if verbose:
            logger_(f"[ADVICE_PLAN] {len(titles)}/{n_total}")

        if len(titles) == prev:
            break

    return titles[:n_total]

def generate_advice(topic: str, n_advice: int = 20,
                    output_dir: str = "outputs",
                    verbose: bool = False,
                    advice_context: str | None = None,
                    advice_titles: list[str] | None = None,
                    file_tag: str | None = None):
    """
    Generate a CSV table of short advices for a given topic.

    advice_context is an optional free-form background used as an epistemic anchor.
    It defines the world model, constraints, hypotheses, and assumptions that guide
    which angles are selected before writing advice items.

    Each advice row contains:
    - ``Num`` (integer index)
    - ``Topic`` (string)
    - ``Advice`` (short title)
    - ``Sentence`` (a spoken line by a character)

    The function prompts the language model to output a strict semicolon-separated
    CSV with exactly these four columns and no extra lines. A lightweight
    post-processing step repairs malformed rows and counts how many advices
    were actually usable.

    Parameters
    ----------
    topic : str
        The input topic that each advice line should refer to.
    n_advice : int, optional
        Number of advice rows requested from the model. Default is 20.
    output_dir : str, optional
        Directory where the CSV output is saved. Default is "outputs".
    advice_context : str or None, optional
        Optional free-form background injected only into the advice generation prompt.
    verbose : bool, optional
        If True, prints additional progress information.

    Returns
    -------
    tuple[str | None, int]
        A pair ``(csv_path, num_rows)`` where ``csv_path`` is the path to the
        generated advice CSV (or ``None`` if generation failed), and ``num_rows``
        is the number of valid advice rows after post-processing.
    """

    if advice_titles is not None:
        titles = [
            " ".join((t or "").strip().split())
            .replace(";", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
            for t in advice_titles
        ]
        titles = [t for t in titles if t]
        if not titles:
            if verbose:
                logger_("!!! Empty advice_titles - skipping step.")
            return None, 0

        n = len(titles)
        width = max(3, len(str(n)))
        ids = [f"A{i:0{width}d}" for i in range(1, n + 1)]

        advice_context_text = (advice_context or "").strip()
        advice_context_block = ""
        if advice_context_text:
            advice_context_block = f"<ADVICE_CONTEXT>\n{advice_context_text}\n</ADVICE_CONTEXT>\n\n"

        def _strip_wrappers(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
            return s.strip()

        def _parse_id_sentence(text_: str) -> dict[str, str]:
            t = _strip_wrappers(text_)
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            if lines and lines[0].casefold().startswith("id;"):
                lines = lines[1:]
            out: dict[str, str] = {}
            for ln in lines:
                ln = re.sub(r"^\s*(?:[-*]|\d+[.)-])\s*", "", ln).strip()
                parts = [p.strip() for p in ln.split(";")]
                if len(parts) < 2:
                    continue
                k = parts[0].split()[0].strip()
                sent = " ".join(" ".join(parts[1:]).replace(";", ",").split())
                sent = sent.replace("\r", " ").replace("\n", " ").strip()
                if k and sent and k not in out:
                    out[k] = sent
            return out

        def _prompt(items: list[tuple[str, str]]) -> str:
            table_in = "\n".join(f"{i};{t}" for i, t in items)
            return f"""{advice_context_block}

Treat ADVICE_CONTEXT as the epistemic anchor: it defines the world model, constraints, hypotheses,
and assumptions that guide which perspectives you select before writing advice items.            

Topic: 
You are given general topic as defined as follows: [{topic}]

Task:
For each row of the input TABLE below, write ONE sentence that is 
1/2) specifically about the given Topic above as theme to keep in mind.
2/2) specifically about the given Advice title in the same row number.

Hard constraints:
- The sentence must mention at least one concrete action element related to the topic (an action verb + a concrete object/context).
- The sentence must be read as a clear and simple sentence which is said by another character to the main one living the scene
- The sentence must reuse at least one non-trivial word from the Advice title (copy it as-is).
- Output ONLY CSV with delimiter ';'
- First line must be: ID;Sentence as the header of the table before the data
- Exactly {len(items)} rows after the header ID;Sentence (written at first row)
- No extra text, no blank lines
- Never use ';' inside sentences

<Format of output table>
Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
⚠️ Output must be ONLY a valid CSV.
⚠️ Output must be WITHOUT EMPTY LINES.
⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
⚠️ Output must be WITHOUT ANY LIST MARKERS.
⚠️ Output must be WITHOUT ANY QUOTES.
⚠️ Output must be WITHOUT ANY BACKTICKS.
⚠️ Output must be WITHOUT ANY MARKDOWN.
⚠️ Output must be WITHOUT ANY INTRODUCTION OR COMMENTARY.
⚠️ Output must be WITHOUT ANY EXTRA TEXT.
⚠️ Output must be WITHOUT ANY PREAMBLE.
⚠️ Output must be ONLY a valid CSV.
⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
⚠️ The CSV must start immediately with this header:
The CSV header is the first line of the generated output as followed, before the line for the values:
Num;Sentence
</Format of output table>

Input TABLE:
ID;Advice
        {table_in}
        """.strip()

        llm = LLMConnect.get_global()
        model = LLMConnect.get_model("ADVICE")

        items_all = list(zip(ids, titles))
        rows: dict[str, str] = {}

        text1 = llm.safe_chat_completion(
            model=model,
            messages=[{"role": "user", "content": _prompt(items_all)}],
            max_tokens=2000,
        ) or ""
        rows.update(_parse_id_sentence(text1))

        missing = [i for i in ids if i not in rows]
        text2 = ""
        if missing:
            items_missing = [(i, titles[ids.index(i)]) for i in missing]
            text2 = llm.safe_chat_completion(
                model=model,
                messages=[{"role": "user", "content": _prompt(items_missing)}],
                max_tokens=1200,
            ) or ""
            rows.update(_parse_id_sentence(text2))

        missing = [i for i in ids if i not in rows or not rows[i].strip()]
        if missing:
            topic_slug = slugify_topic(file_tag or topic)
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, f"bad_rows_{topic_slug}_advice_id.log")
            with open(log_path, "a", encoding="utf-8", newline="") as f:
                f.write("MISSING_IDS\n")
                for m in missing:
                    f.write(m + "\n")
                f.write("\nRAW1\n" + _strip_wrappers(text1) + "\n")
                f.write("\nRAW2\n" + _strip_wrappers(text2) + "\n\n")
            return None, 0

        df = pd.DataFrame({
            "Num": list(range(1, n + 1)),
            "Topic": [topic] * n,
            "Advice": titles,
            "Sentence": [rows[i].replace(";", ",").strip() for i in ids],
        })

        topic_slug = slugify_topic(file_tag or topic)
        os.makedirs(output_dir, exist_ok=True)
        path_csv = os.path.join(output_dir, f"Advice_{topic_slug}.csv")
        with open(path_csv, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, sep=";", index=False, lineterminator="\n")
        return path_csv, len(df)

    # -----------------------------------------------------------------
    advice_context_text = (advice_context or "").strip()
    advice_context_block = ""
    if advice_context_text:
        advice_context_block = f"""

    <ADVICE_CONTEXT>
    {advice_context_text}
    </ADVICE_CONTEXT>

    Treat ADVICE_CONTEXT as the epistemic anchor: it defines the world model, constraints, hypotheses,
    and assumptions that guide which perspectives you select before writing advice items.
    """

    prompt = f"""
    Generate {n_advice} safety or behavioral advices for the topic: {topic}.
    {advice_context_block}

    # <PARTICULAR_CASE_QUERY_INSTEAD_OF_TOPIC_FOR_ADVICE>
    # If the topic is phrased as a general question, treat each item as a possible
    # answer or perspective on that question.
    # Otherwise, treat each item as a behavioral advice related to the topic.
    # </PARTICULAR_CASE_QUERY_INSTEAD_OF_TOPIC_FOR_ADVICE>

    For each advice, include:
    - A short title (3-6 words)
    - A clear, simple sentence which is said by another character to the main one living the scene

    # <Format_Outline>
    # For each row:
    # - In the "Topic" column: repeat exactly the topic text.
    # - In the "Advice" column: write a short title (3-6 words) describing the
    #   angle or perspective (e.g. "Doctor's view", "Practical approach",
    #   "Close friend's advice").
    # - In the "Sentence" column: write one clear, simple sentence that is spoken
    #   by a character and that states the advice or answer in natural language.
    # </Format_Outline>

    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ Output must be WITHOUT ANY LIST MARKERS.
    ⚠️ Output must be WITHOUT ANY QUOTES.
    ⚠️ Output must be WITHOUT ANY BACKTICKS.
    ⚠️ Output must be WITHOUT ANY MARKDOWN.
    ⚠️ Output must be WITHOUT ANY INTRODUCTION OR COMMENTARY.
    ⚠️ Output must be WITHOUT ANY EXTRA TEXT.
    ⚠️ Output must be WITHOUT ANY PREAMBLE.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Topic;Advice;Sentence
    </Format>
    """
    text = LLMConnect.get_global().safe_chat_completion(model=LLMConnect.get_model("ADVICE"), 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000)
    topic_slug = slugify_topic(topic)
    if not text:
        if verbose: logger_("!!! No answer from remote model - skipping step.")
        return None, 0
    log_path     = os.path.join(output_dir, f"bad_rows_{topic_slug}_advice.log")
    text         = postprocess_csv_text_basic(text,expected_fields=4,log_path=log_path,verbose=verbose)
    file, real_n = save_output(text, f"Advice_{topic_slug}", output_dir)
    try:
        df = pd.read_csv(file, sep=";")
        if list(df.columns) != ["Num", "Topic", "Advice", "Sentence"]:
            if verbose: logger_("!! CSV header unexpected: %s", df.columns.tolist())
        real_n = len(df)
    except Exception:
        real_n = 0

    return file, real_n

def generate_mapping(advice_file: str,
                     SN_file: str,
                     DE_file: str,
                     output_dir: str = "outputs",
                     verbose: bool = False):
    """
    Generate a Mapping CSV that assigns SN/DE codes to each advice row.

    The function reads:
    - the advice CSV (with a ``Num`` column),
    - the official SN reference table,
    - the official DE reference table,

    and prompts the model to output a strict semicolon-separated CSV with
    exactly three columns::

        Num;Code_SN;Code_DE

    A post-processing step fixes basic CSV issues and logs malformed rows.

    Parameters
    ----------
    advice_file : str
        Path to the source Advice CSV whose rows must be annotated.
    llm : LLMConnect
        Unified LLM wrapper used to call the underlying model(s).
    SN_file : str
        Path to the official SN reference CSV.
    DE_file : str
        Path to the official DE reference CSV.
    model : str, optional
        Model name passed to ``llm.safe_chat_completion``. Default is "gpt-4o-mini".
    output_dir : str, optional
        Directory where the Mapping CSV and logs are written. Default is "outputs".
    verbose : bool, optional
        If True, prints basic progress information.

    Returns
    -------
    tuple[str | None, int]
        A pair ``(csv_path, num_rows)`` where ``csv_path`` is the path to the
        generated Mapping CSV (or ``None`` if generation failed), and
        ``num_rows`` is the number of valid rows after post-processing.
    """

    adv = pd.read_csv(advice_file, sep=";")
    prompt = f"""
    You are given a table of advices. For each advice of number Num, assign:
    - Num number as first column
    - A narrative structure (SN code, e.g., SN1, SN3c…) as second column
    - An emotional dynamic (DE code, e.g., DE1, DE7…) as third column

    <Rules FOR DIVERSITY>
    It is mandatory to follow this rules:
    - You are mapping narrative advices to corresponding SN (Narrative Structure)
      and DE (Emotional Dynamic) codes.
    - Adopt a *diverse and pedagogical perspective* on the theme.
    - Encourage variation and contrast between lines: 
      ->some mappings should emphasize introspection, others analysis, 
      ->some emotional, others explanatory or didactic.
      ->Avoid using the same SN or DE too often. 
      ->Across all items, aim for a wide variety of narrative and emotional pairings
      Remember: diversity is part of the evaluation criteria of this mapping task.
    </Rules FOR DIVERSITY>

    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ The CSV must contain exactly 3 columns (for column names= Num;Code_SN;Code_DE), no more no less.
    ⚠️ Only use SN and DE CODES from the provided reference lists.
    ⚠️ ALWAYS write the CODE only (e.g. "SN3", "DE2") in columns Code_SN and Code_DE.
    ⚠️ NEVER write the SN/DE names or definitions in Code_SN or Code_DE.
       - FORBIDDEN examples: "Problem Prevention", "Joyful Serenity",
         "Calm neutral baseline", complete sentences, or any description.
       - ONLY allowed: raw codes such as "SN1", "SN3c", "DE4", etc.
    ⚠️ Do not invent new codes for SN et DE: SELECT ONLY CODES AVAILABLE.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ If you need to separate items inside a cell, use character '_'  or spaces ' ' ONLY, AND NEVER ";" or ",".
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Code_SN;Code_DE
    </Format>

    <OFFICIAL_CODES_SN_DE>
    The EXISTING AND ONLY ALLOWED SN and DE are defined as below.
    """
    sn_ref = pd.read_csv(SN_file, sep=";")
    de_ref = pd.read_csv(DE_file, sep=";")
    
    prompt += "\n"
    prompt += "\n<OFFICIAL_SN_CODES>\n"
    prompt += "\nHere just below will be a table which gives the SN codes, the SN names and the SN structure definition.\n"
    prompt += "\nThe full list of ONLY ALLOWED SN codes is to choose from:\n"    
    prompt += sn_ref.to_csv(sep=";", index=False)
    prompt += "\n</OFFICIAL_SN_CODES>\n"
    prompt += "\n"
    prompt += "\n<OFFICIAL_DE_CODES>\n"
    prompt += "\nHere just below will be a table which gives the DE codes, the DE names and the DE structure definition.\n"
    prompt += "\nThe full list of ONLY ALLOWED DE codes is to choose from:\n"    
    prompt += de_ref.to_csv(sep=";", index=False)
    prompt += "\n</OFFICIAL_DE_CODES>\n"
    prompt += "\n"
    prompt += "\nWhen choosing a SN code and DE code, think enough deeply by reading the name and the definition to help the best selection for the advice.\n"
    prompt += "\n⚠️ You must ONLY select codes from the official list below. \n"
    # prompt += "\n⚠️ If you cannot find a suitable code, choose SN1 and DE1 by default. \n"
    prompt += "\n⚠️ If you cannot find a suitable code, choose the closest allowed DE with suitable meaning; avoid defaulting to DE1 or any other DE.\n"
    prompt += "\n⚠️ Never invent new codes (like SNk or DEk where k is not relevant).\n"
    prompt += "</OFFICIAL_CODES_SN_DE>"
    
    text = LLMConnect.get_global().safe_chat_completion(model=LLMConnect.get_model("MAPPING"), 
        messages=[{"role": "user", "content": prompt + "\n\n" + adv.to_csv(sep=';', index=False)}],
)
    if not text:
        if verbose: logger_("!!! No answer from remote model - skipping step.")
        return None, 0
    
    log_path = os.path.join(
        output_dir,
        f"bad_rows_{Path(advice_file).stem.replace('Advice', 'Mapping')}.log"
    )
    text = postprocess_csv_text_basic(text,expected_fields=3,log_path=log_path,verbose=verbose)
    
    file, real_n = save_output(
        text,
        os.path.basename(advice_file).replace("Advice", "Mapping").replace(".csv", ""),
        output_dir
    )
    try:
        df = pd.read_csv(file, sep=";")
        if list(df.columns) != ["Num", "Code_SN", "Code_DE"]:
            if verbose: logger_("!! Mapping CSV header unexpected: %s", df.columns.tolist())
        real_n = len(df)
    except Exception as e:
        if verbose: logger_(f"!! Failed to parse mapping file {file}")
        real_n = 0

    return file, real_n


def generate_context(advice_file: str,
                     output_dir: str = "outputs",
                     verbose: bool = False,
                     context_context: str | None = None):
    """
    Generate a CSV file with narrative context for each advice row.

    For every ``Num`` in the advice CSV, the model is asked to produce a
    compact scene description with fields such as:

    - ``Presence`` (who is around, including the speaker)
    - ``Location`` (setting)
    - ``Sensation`` (noise, light, smell, etc.)
    - ``Time`` (time of day)
    - ``Moment`` (atmosphere / mood keyword)
    - ``First_Name`` (plausible name for the main character)

    The result is a semicolon-separated CSV aligned on ``Num`` and repaired by
    a basic CSV post-processor.

    Parameters
    ----------
    advice_file : str
        Path to the Advice CSV that provides the ``Num`` and advice texts.
    llm : LLMConnect
        Unified LLM wrapper used to call the underlying model(s).
    model : str, optional
        Model name passed to ``llm.safe_chat_completion``. Default is "gpt-4o-mini".
    output_dir : str, optional
        Directory where the context CSV and logs are written. Default is "outputs".
    context_context : str or None, optional
        Optional free-form background injected only into the context generation prompt.
    verbose : bool, optional
        If True, prints basic progress information.

    Returns
    -------
    tuple[str | None, int]
        A pair ``(csv_path, num_rows)`` where ``csv_path`` is the path to the
        generated context CSV (or ``None`` if generation failed), and
        ``num_rows`` is the number of valid rows after post-processing.
    """

    context_context_text = (context_context or "").strip()
    context_context_block = ""
    if context_context_text:
        context_context_block = f"""

    <CONTEXT_EXTRA_PROMPT>
    {context_context_text}
    </CONTEXT_EXTRA_PROMPT>

    Treat CONTEXT_EXTRA_PROMPT as the epistemic anchor: it defines the world model, constraints, 
    hypotheses, and assumptions that guide which perspectives you select before writing advice items.
    """

    adv = pd.read_csv(advice_file, sep=";")
    prompt = f"""
    You are going to generate a CSV table of contextual information for textual generation of advice.
    Before the definiong of the precise required mandatory format and contents of the table, memorize this.
    {context_context_block}

    For each advice in the table with columns (CSV), generate narrative context details:
    <Contents>
    - Character (age, role, gender) mais SANS PRENOM (aka First_Name), pour ne pas risquer d'incohérence avec colonne ci-après
    - Presence = all the people around who are present around the Character and between parenthesis explicitly who speaks the advice. Examples:
        "crow (alone, aka inner voice)"
        "street walkers (with an older woman who advises)"
        "store byers (with his daughter, who gives the advice)"
        "teacher classmates (with a classmat, who speaks)"
        "people in street (with near/aside/behind/afront a stranger who speaks the advice)"
        "someone at a window (with near/aside/behind/afront a stranger who speaks the advice)"
        "people on the sidewalk (with near/aside/behind/afront a stranger who speaks the advice)"
    - Location (urban, school, park…)
    - Sensation (noise, smell, light…)
    - Time (morning, afternoon…)
    - Moment (mood ambiance in one word like among [clear,sunny,cloudy,bright,calm,cool,warm,windy,quiet,noisy,soft,vivid,lively,nighty,peaceful,saturared] )
    - First_Name (plausible)
    </Contents>
    
    <Format>
    Output must be ONLY a valid CSV, without line without header and without line after the valid CSV contents.
    ⚠️ The CSV must contain exactly 8 columns (for column names= Num;Character;Presence;Location;Sensation;Time;Moment;First_Name), no more no less.
    ⚠️ Output must be ONLY a valid CSV.
    ⚠️ Output must use standard UTF-8 characters (no special symbols or emojis).
    ⚠️ Output must be WITHOUT EMPTY LINES.
    ⚠️ Output must be WITHOUT additional text, WITHOUT explanation, WITHOUT code blocks.
    ⚠️ Never use semicolons inside the cells (they are only separators).
    ⚠️ If you need to separate items inside a cell, use character '_'  or spaces ' ' ONLY, AND NEVER ";" or ",".
    ⚠️ Inside cells of the csv after header: Never insert any semicolons which are used as csv separators
    ⚠️ Inside cells of the csv after header: Use only plain text separated by spaces or between parenthesis.
    ⚠️ The Character field is with three fields: precise age, precise role, and precise gender.
    ⚠️ The First_Name field must contain ONLY the given VALID name, without any description.
    ⚠️ The Presence field is as following ONLY FORMAT "<Group or Person around the Character> (with <Speaker> who advises/talks)" in two parts
    ⚠️ DIVERSITY RULES (MANDATORY) --
    • First_Name MUST be varied across all rows in the CSV. No repetition of the same name is allowed.
    • Each Character must have distinct gender/role/age combinations where possible.
    • If a name would repeat, choose another realistic one from common French or English first names.
    • Vary Location, Time and Moment to ensure each line feels unique.
    • Never reuse the same combination of Character + First_Name twice.
    ⚠️ The CSV must start immediately with this header:
    The CSV header is the first line of the generated output as followed, before the line for the values:
    Num;Character;Presence;Location;Sensation;Time;Moment;First_Name
    </Format>
    """
    text = LLMConnect.get_global().safe_chat_completion(model=LLMConnect.get_model("CONTEXT"),
        messages=[{"role": "user", "content": prompt + "\n\n" + adv.to_csv(sep=';', index=False)}],)
    if not text:
        if verbose: logger_("!!! No answer from remote model - skipping step.")
        return None, 0
    log_path = os.path.join(output_dir,
    f"bad_rows_{Path(advice_file).stem.replace('Advice', 'Context')}.log")
    text = postprocess_csv_text_basic(text,expected_fields=8,log_path=log_path,verbose=verbose)   

    file, real_n = save_output(
        text,
        os.path.basename(advice_file).replace("Advice", "Context").replace(".csv", ""),
        output_dir
    )
    try:
        df = pd.read_csv(file, sep=";")
        expected_cols = ["Num", "Character", "Presence", "Location", "Sensation", "Time", "Moment", "First_Name"]
        if list(df.columns) != expected_cols:
            if verbose: logger_("!! Context CSV header unexpected: %s", df.columns.tolist())
        real_n = len(df)
    except Exception as e:
        if verbose: logger_(f"!! Failed to parse context file {file}")
        real_n = 0

    return file, real_n
