# <br>
# **<u>NarrEmGen: Narrative Generation Pipeline (CLI & GUI)</u>**<br>
**<u>(Implementing Partially The SN/DE/K Method For a Controlled Generation)</u>**<br>
**<u>(i.e. Generating Full Booklets of Advice or Answers from a Topic or a Question)</u>**<br>
<br>

## Main modules of narremgen

- `pipeline`: Entry point for batch generation, variants, stats, and exports per topic run.
- `llmcore`: Unified LLM router (role→model mapping, retries, multi-provider support).
- `data`: Input preparation and CSV handling for topic–advice–prompt-based generation.
- `narratives`: Text post-processing, style control, and SN/DE-aware narrative realization.
- `variants`: Planning and batch rewriting into alternative styles (direct, formal, etc.) with stats.
- `themes`: LLM-based theme discovery and assignment for advice corpora, producing themes+assignments.
- `chapters`: Build chaptered corpora (CSV/JSON) from themes or manual grouping, for book-like exports.
- `export`: Plain-text and LaTeX exporters (merged `.txt` and `book_*.tex` from neutral and variants).
- `analyzestats`: Length, lexical, emotion and SN/DE distribution analysis, with CSV summaries and plots.
- `utils`: Shared helpers for workdirs, filenames, CSV repair, backups, and neutral corpus construction.
- `gui`: Optional Tkinter GUI for generation, or readings aligned/selected texts, or segmentation.
- `main`: Optional command-line terminal module for the generation with input arguments.

    
## Key features

- Generation of a <u>Corpus</u> of <u>Stories</u> (of varying and controlled structures) and <u>Formal Texts</u> for advice from a topic (full sentence).
- Multi-batch narrative pipeline using a configurable LLM router (`llmcore`) across several providers with a command-line interface.
- Automatic topic and advice mapping, SN/DE-structured neutral generation, and aligned variant rewriting (direct, formal, other styles).
- Robust CSV workflow: filtering, renumbering, safe merging of advice/sentence/context/mapping, consistent filenames, variant workdirs.
- LLM-driven theme extraction and assignment, plus chapter construction for organizing texts into coherent sections (classes of texts).
- Plain-text and TeX export of neutral and variant corpora (merged narrative files and full chaptered books for text reading/selection).
- Integrated corpus analysis: lexical richness, length, emotion profiles, and SN/DE distributions, including neutral vs. variant comparison.
- Textual statistics and emotion statistics of specialized language models from the literature for evaluation of generated texts or corpus.
- Ready-to-use structure for reproducible experiments in text generation with emotions for character and educational content synthesis.
- Graphical user interface for generation with api key checkings, creation of variants, and reading/selection of aligned textes for a topic.
- Available connection to OpenAI, OpenRouter, Google-GenAI, Mistral, etc for text generation (see python code and interface for dry-run).
- No limited length for topic str, available command for adding file/str long text as context for advice or generation stages in pipeline.

Note: This package is provided *“as is”* for the research and educational purposes.  <br>
      The code was written/debogged in iterative way with help of gpt5 openai + vs code. <br>
      All texts generated are synthetic and intended for future experimentations only. <br>
      Last version in directory package for pypi. 

To do: improve genericity, generality and robutness, add parallelism, classes re-factor.

## Usage

```
pip install narremgen
```

```python
import narremgen
from narremgen import pipeline

run_pipeline(
    topic="Walking_in_the_city",
    output_dir="./outputs",
    assets_dir="./narremgen/settings",
    n_batches=2,
    n_per_batch=20,
    output_format="txt",
    verbose=False
)
```

With command lines in the terminal: Generation <br>

```python
# Pipeline + variants (default, not user ones)
python -m narremgen.main \
  --topic "Walking_in_the_city" \
  --output-dir "./outputs" \
  --batches 2 \
  --per-batch 20 \
  --output-format txt \
  --verbose
```

```python
# Pipeline without variants (neutral only)
python -m narremgen.main \
  --topic "Walking_in_the_city" \
  --output-dir "./outputs" \
  --batches 2 \
  --per-batch 20 \
  --output-format txt \
  --skip-variants \
  --verbose
```

```python
# Dry-test without generation pipeline
python -m narremgen.main \
  --diagnostic-dry-run \
  --verbose
```

With command lines in the terminal: GUI <br>

```python
# Interface generation+reading+saving
python -m narremgen.gui
```

## Examples of output datasets

| ID | Corpus | n_texts |
|----|---------|---------|
| D1 | Keeping_town_clean | 257 |
| D2 | Learning_mistakes | 279 |
| D3 | Learning_skills | 296 |
| D4 | Protecting_water_forests | 253 |
| D5 | Stay_healthy | 300 |
| D6 | Walk_city | 276 |
| D7 | Walk_dawn | 271 |
| D8 | Walk_rain | 296 |
| D9 | Walk_water | 213 |
| D10 | Walk_wild | 255 |

Each generated corpus is stored under `outputs/` in CSV and TXT format.  
The naming convention is: `outputs/<corpus_name>_1/` for its directory<br>
<br>
Each directory contains:
```
topic, advice, and mapping tables in csv format and generated texts
and two subdirectories containing generated batched texts + csv files
plus directories for variants with statistics + chaptered tex files
```

## Warning

Only informed users or trainers should use this system in practice. <br>
Some advice may be missing or mistaken du to ia/programming. <br>
In future automatic checkings may be implemented for end user.

## References

- Priam, Rodolphe (2025). *Narrative and Emotional Structures For Generation Of Short Texts For Advice.*, hal-05135171, 2025.

---

© NarrEmGen Project, 2025-2026.
     
