"""
narremgen.data
=============
LLM-driven generation of the core Advice, Context, and Mapping CSV tables.

Wraps LLMConnect calls and post-processing to generate clean,
semicolon-separated datasets with stable naming conventions, forming the
entry point of the Narremgen pipeline.
"""

import os
import pandas as pd
from pathlib import Path
from .llmcore import LLMConnect
from .utils import save_output
from .utils import slugify_topic
from .utils import postprocess_csv_text_basic

import logging
logger = logging.getLogger(__name__)
logger_ = logger.info


def generate_advice(topic: str, n_advice: int = 20,
                    output_dir: str = "outputs",
                    verbose: bool = False,
                    advice_context: str | None = None):
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
    verbose : bool, optional
        If True, prints additional progress information.

    Returns
    -------
    tuple[str | None, int]
        A pair ``(csv_path, num_rows)`` where ``csv_path`` is the path to the
        generated advice CSV (or ``None`` if generation failed), and ``num_rows``
        is the number of valid advice rows after post-processing.
    """

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
                     verbose: bool = False):
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
    verbose : bool, optional
        If True, prints basic progress information.

    Returns
    -------
    tuple[str | None, int]
        A pair ``(csv_path, num_rows)`` where ``csv_path`` is the path to the
        generated context CSV (or ``None`` if generation failed), and
        ``num_rows`` is the number of valid rows after post-processing.
    """

    adv = pd.read_csv(advice_file, sep=";")
    prompt = f"""
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
