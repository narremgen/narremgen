"""
narremgen.main_narragem_full
============================
Entry point for:
1) Full narremgen generation pipeline (advice / context / mapping / narratives)
2) Variants pipeline (chapters + style variants)
"""

import os
import json
import traceback
from pathlib import Path
import pandas as pd


from narremgen.narremgen.utils import build_csv_name
from narremgen.narremgen.utils import slugify_topic
from narremgen.narremgen.core import UnifiedLLM#, dry_run_model
from narremgen.narremgen.pipeline import run_pipeline
# from narremgen.narremgen.utils import find_last_workdir
from narremgen.narremgen.utils import get_workdir_for_topic
from narremgen.narremgen.variants import run_one_variant_pipeline
from narremgen.narremgen.themes import run_llm_theme_pipeline

LIST_VARIANTS = ["simple", "formal", "naive", "algebraic"]

RUN_NEUTRAL_PIPELINE  = True
RUN_VARIANT_PIPELINE  = True
DO_CLASSIF            = True
VERBOSE               = True


def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(SCRIPT_DIR, "key.txt"), encoding="utf-8") as f:
        api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

    with open(os.path.join(SCRIPT_DIR, "key2.txt"), encoding="utf-8") as f:
        api_key2 = f.read().strip()
    os.environ["MISTRAL_API_KEY"] = api_key2

    with open(os.path.join(SCRIPT_DIR, "key3.txt"), encoding="utf-8") as f:
        api_key3 = f.read().strip()
    os.environ["GEMINI_API_KEY"] = api_key3
    
    PACKAGE_DIR = os.path.join(SCRIPT_DIR, "narremgen/narremgen/")
    ASSETS_DIR = os.path.join(SCRIPT_DIR, "narremgen/settings")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

    if VERBOSE:
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Assets directory: {ASSETS_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
    
    llm_models = {
        "ADVICE":             "openai\\gpt-4o-mini",#"ollama\\phi3-chat:latest",
        "MAPPING":            "openai\\gpt-4.1", #"openai\\gpt-4o"
        "CONTEXT":            "openai\\gpt-4o-mini",
        "NARRATIVE":          "openai\\gpt-4o-mini",
        "THEME_ANALYSIS":     "openai\\gpt-4.1-mini", #"ollama\\phi3-chat:latest", #
        "VARIANTS_GENERATION":"openai\\gpt-4o-mini",#"ollama/mistral-chat:latest"
    }
    
    UnifiedLLM.init_global(
        llmodels=llm_models,
        default_model="openai\\gpt-4o-mini",
        temperature=0.7,
        max_tokens=1024,
        request_timeout=600,
    )
    
    llm = UnifiedLLM.get_global()

    topic_all  = "how do healthy walk at the small city in winter with rain and cold"
    #topic_dir = slugify_topic(topic_all, mode="old")
    topic_dir = slugify_topic(topic_all)

    if VERBOSE: 
        print(f"Launching pipeline for {topic_all}")

    WORKDIR = get_workdir_for_topic(
        OUTPUT_DIR,
        topic_dir,
        create_new=RUN_NEUTRAL_PIPELINE,
    )

    if VERBOSE:
        print(f"[VARIANTS] Using WORKDIR = {WORKDIR}")
    # -------------------------------------------------------------------------
    # NEUTRAL GENERATION
    # -------------------------------------------------------------------------

    if RUN_NEUTRAL_PIPELINE:
        final_file = run_pipeline(
                        topic=topic_all,
                        workdir=WORKDIR,
                        assets_dir=ASSETS_DIR,
                        n_batches=1,
                        n_per_batch=20,
                        output_format="txt",        
                        verbose=VERBOSE,
                        )
    else:
        if VERBOSE:
            print("[INFO] Skipping neutral generation, reuse existing WORKDIR.")
    
   
    # -------------------------------------------------------------------------
    # VARIANTS GENERATION
    # -------------------------------------------------------------------------

    SEED = 1234567
    RANDOM_SEED = 123

    SIMPLE_PROMPT_FILE = Path(SCRIPT_DIR) / "simple_prompt_variants.txt"
    FORMAL_PROMPT_FILE = Path(SCRIPT_DIR) / "simple_prompt_variants_formal.txt"

    SIMPLE_MAX_TOKENS = 3000
    BATCH_SIZE        = 10

    OVERWRITE_EXISTING = False

    algebra_path = {"sn_pathdf": Path(ASSETS_DIR) / "SN_extended.csv",
                       "de_pathdf": Path(ASSETS_DIR) / "DE.csv",
                       "ki_pathdf": WORKDIR / build_csv_name("context",
                                                     "FilteredRenumeroted",
                                                    WORKDIR.name.rsplit("_", 1)[0],),
                       "opstc_pathdf": Path(ASSETS_DIR) / "operators_structural.csv",
                       "opstl_pathdf": Path(ASSETS_DIR) / "operators_stylistic.csv"}

    if RUN_VARIANT_PIPELINE:
        for varianttype in LIST_VARIANTS:
            if VERBOSE: print(f"=== MODE {varianttype.upper()} (batch rewrite) ACTIVATED ===")

            prompt_file = SIMPLE_PROMPT_FILE
            if varianttype == "formal":
                prompt_file = FORMAL_PROMPT_FILE

            try:
                outvariant = run_one_variant_pipeline(
                    WORKDIR=WORKDIR,
                    VARIANT_TYPE=varianttype, 
                    algebra_path=algebra_path,
                    SIMPLE_PROMPT_FILE=prompt_file,
                    SIMPLE_MAX_TOKENS=SIMPLE_MAX_TOKENS,
                    BATCH_SIZE=BATCH_SIZE,
                    RANDOM_SEED=RANDOM_SEED,                
                    VERBOSE=VERBOSE,
                    overwrite_existing=OVERWRITE_EXISTING,
                    compute_textstats=True,
                )
            except Exception:
                print("[ERREUR FATALE]")
                print(traceback.format_exc())

    # -------------------------------------------------------------------------
    # OPTIONAL THEMES GENERATION FROM MAIN TOPIC
    # -------------------------------------------------------------------------

    if DO_CLASSIF:
        THEMES_JSON_NAME          = "advice_groups_llm.json"
        THEMES_ASSIGNMENT_JSON_NAME = "advice2groups_llm.json"
        themes_assign_path = WORKDIR / THEMES_ASSIGNMENT_JSON_NAME
        if not themes_assign_path.exists():
            if VERBOSE:
                print(
                    f"[INFO] Aucun fichier {themes_assign_path.name} trouvé, "
                    "lancement du pipeline LLM de thèmes..."
                )
        run_llm_theme_pipeline(
            workdir=WORKDIR,
            themes_json_name=THEMES_JSON_NAME,
            assignments_json_name=THEMES_ASSIGNMENT_JSON_NAME,
            n_themes_min=7,
            n_themes_max=12,
            batch_size=40,
            verbose=VERBOSE,
        )

if __name__ == "__main__":
    main()
