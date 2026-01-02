"""
narremgen.gui
=============
Tkinter-based GUI for Narremgen. This module proposes three main interfaces:
1) A workbench for loading, aligning, filtering, and selecting narrative texts
   across up to three synchronized views (SN/DE code filters, search, stats,
   gauges, batch export).
2) A generation panel that launches Narremgen pipelines (neutral, variants,
   full), manages LLM settings, variant definitions, assets paths, logging,
   and run metadata snapshots.
3) Segmenter to take a text and divide into segments automatically,
   or with the keyboard or the mouse and finally store the segments.

The module combines:
- a data/model layer (MergerModel) handling alignment, filters, search,
  selection, and statistics,
- multiple UI panels (NavigationPanel, VariantView, StatsPanel, Dashboard),
- a controller linking model <-> views,
- a main window embedding the Workbench and Generator tabs.

In short: a compact GUI front-end that unifies corpus inspection and LLM-based
generation workflows for Narremgen.
"""

from tkinter import scrolledtext, messagebox
from tkinter import filedialog, ttk
import tkinter.font as tkfont
import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import Counter
from dataclasses import dataclass, asdict
import re, os, shutil, json, configparser
import xml.etree.ElementTree as ET
import logging, traceback, hashlib
import threading, atexit, queue, sys
import subprocess, contextlib
from datetime import datetime
from pathlib import Path
import pandas as pd
from contextlib import ExitStack
import importlib.resources as pkgres
from datetime import datetime, timezone
import time
from .pipeline import run_pipeline
from .variants import run_one_variant_pipeline
from .llmcore import LLMConnect, LLMCancellation
from .utils import slugify_topic, get_workdir_for_topic
from .utils import load_neutral_data, build_csv_name
from .themes import run_llm_theme_pipeline
from .chapters import create_chapters_from_theme_assignments
from .export import build_merged_tex_from_csv
from .analyzestats import build_global_stats_table
from .segmenter import SimpleSegmenterApp
import logging
_logger = logging.getLogger("narremgen.gui")
# logger = logging.getLogger(__name__)
# logger_ = logger.info

FILE_DIR = Path(__file__).resolve().parent

def _user_config_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "narremgen"
    return Path.home() / ".narremgen"

def _ensure_ini_from_assets(dst: Path, assets_dir: Path, filename: str):
    if dst.exists():
        return
    src = assets_dir / filename
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

@contextlib.contextmanager
def _interprocess_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

def _atomic_write_ini(path: Path, write_fn):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

CONFIG_ROOT = _user_config_root()
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = CONFIG_ROOT / "narremgen_gui.ini"
VARIANTS_CONFIG_PATH = CONFIG_ROOT / "narremgen_variants.ini"
UI_CONFIG_PATH = CONFIG_ROOT / "narremgen_ui.ini"

DEFAULT_OUTPUT_DIR = Path.home() / ".narremgen" / "outputs"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ASSETS_STACK = ExitStack()
atexit.register(_ASSETS_STACK.close)

def _packaged_assets_dir() -> Path:
    res = pkgres.files("narremgen").joinpath("settings")
    return _ASSETS_STACK.enter_context(pkgres.as_file(res))
try:
    DEFAULT_ASSETS_DIR = _packaged_assets_dir()
except Exception:
    DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent / "settings"

LOCK_PATH = CONFIG_ROOT / ".lock"

_ensure_ini_from_assets(CONFIG_PATH, DEFAULT_ASSETS_DIR, "narremgen_gui.ini")
_ensure_ini_from_assets(VARIANTS_CONFIG_PATH, DEFAULT_ASSETS_DIR, "narremgen_variants.ini")
_ensure_ini_from_assets(UI_CONFIG_PATH, DEFAULT_ASSETS_DIR, "narremgen_ui.ini")

DEFAULT_NEUTRAL_N_BATCHES = 15
DEFAULT_NEUTRAL_N_PER_BATCH = 20
DEFAULT_VARIANT_BATCH_SIZE = 20
DEFAULT_SIMPLE_MAX_TOKENS = 3000

DEFAULT_THEMES_MIN = 7
DEFAULT_THEMES_MAX = 12
DEFAULT_THEMES_BATCH_SIZE = 20

DEFAULT_LLM_MODELS = {
    "ADVICE": "openai\\gpt-4o-mini",
    "MAPPING": "openai\\gpt-4.1",
    "CONTEXT": "openai\\gpt-4o-mini",
    "NARRATIVE": "openai\\gpt-4o-mini",
    "THEME_ANALYSIS": "openai\\gpt-4.1-mini",
    "VARIANTS_GENERATION": "openai\\gpt-4o-mini",
}

REQUIRED_ASSET_FILES = (
    'SN_extended_raw.csv',
    'SN.csv',
    'DE.csv',
    'operators_structural_raw.csv',
    'operators_stylistic_raw.csv',
)


KEYFILE_BASENAME = "llmkey_narremgen.txt"

def _split_provider(model_spec: str) -> tuple[str, str]:
    s = (model_spec or "").strip()
    if not s:
        return ("", "")
    for sep in ("\\", "/", ":"):
        if sep in s:
            p, m = s.split(sep, 1)
            return (p.strip().lower(), m.strip())
    return ("", s)

def _required_env_for_provider(provider: str) -> list[list[str]]:
    p = (provider or "").lower().strip()
    if p == "openai":
        return [["OPENAI_API_KEY"]]
    if p == "mistral":
        return [["MISTRAL_API_KEY"]]
    if p == "openrouter":
        return [["OPENROUTER_API_KEY"]] #use for Gemini, etc
    if p in ("grok", "xai"):
        return [["XAI_API_KEY", "GROK_API_KEY"]]
    if p == "huggingface":
        return [["HUGGINGFACE_API_KEY", "HF_API_KEY"]]
    if p == "gemini":
        return [["GEMINI_API_KEY", "GOOGLE_API_KEY"]]
    if p == "anthropic":
        return [["ANTHROPIC_API_KEY"]]
    if p == "ollama":                  #use for Phi3, etc
        return []  # local
    return []

KNOWN_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "grok": ["XAI_API_KEY", "GROK_API_KEY"],
    "xai": ["XAI_API_KEY", "GROK_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_API_KEY"],
    "ollama": [],
    "anthropic": ["ANTHROPIC_API_KEY"],
}

FONT_SIZE_CHOICES: tuple[int, ...] = (10, 11, 12, 13, 14, 16, 18, 20)
MAX_LOG_MSG_PER_TICK = 200

def _providers_from_llm_models(llm_models: Dict[str, str]) -> dict[str, str]:
    providers: dict[str, str] = {}
    for role, spec in (llm_models or {}).items():
        if not spec:
            continue
        provider, _model = _split_provider(str(spec))
        if provider:
            providers[role] = provider
    return providers

@dataclass(frozen=True)
class RunContext:
    topic: str
    workdir: Path
    mode: str
    selected_variant: Optional[str]
    list_variant_data: Tuple[Dict[str, Any], ...] = ()


class LogRedirector:

    def __init__(self, q):
        self.q = q

    def write(self, text):
        text = str(text)
        if text.strip():
            self.q.put(text)

    def flush(self):
        pass

class SensitiveFormatter(logging.Formatter):
    """Formatter to remove secrets."""
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        return LLMConnect._scrub_secret(s)

class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__(level=logging.INFO)
        self.q = q
        self.setFormatter(SensitiveFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            self.handleError(record)

class RunCancelled(RuntimeError):
    """Raised to abort a run gracefully."""


class ConfigManager:
    def __init__(self, filename=None):
        target = CONFIG_ROOT / "narremgen_ui.ini"
        self.filename = str(target if filename is None else filename)
        self.config = configparser.ConfigParser()
        self.defaults = {
            'UI': {
                'window_size': '1600x900',
                'font_family': 'Consolas',
                'font_size_text': '11',
                'font_size_list': '11',
                'list_row_height': '24',
                'font_size_variants': '11',
                'font_size_nav': '11',
                'font_size_stats': '11',
                'font_size_logs': '11',
                'ui_font_size_main': '11',
                'ui_font_size_variants': '11',
                'ui_font_size_nav': '11'
            },
            'COLORS': {
                'bg_selected': '#cce5ff',
                'bg_nav_selected': '#4caf50',
                'fg_nav_selected': 'white',
                'fg_highlight': 'blue',
                'header_bg': '#e0e0e0'
            },
            'PATHS': {
                'last_output_dir': os.getcwd(),
                'last_assets_dir': os.getcwd(),
                'last_workdir': os.getcwd()
            }
        }
        self.load()

    def load(self):
        if not os.path.exists(self.filename):
            self._create_default()
        try:
            self.config.read(self.filename)
        except (OSError, configparser.Error):
            _logger.warning("Config read failed; fallback to defaults.")
            self._create_default()

    def _create_default(self):
        for section, options in self.defaults.items():
            self.config[section] = options
        with _interprocess_lock(LOCK_PATH):
            _atomic_write_ini(Path(self.filename), self.config.write)

    def set(self, section, key, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))
        with _interprocess_lock(LOCK_PATH):
            _atomic_write_ini(Path(self.filename), self.config.write)

    def get(self, section, key, fallback=None):
        try:
            if self.config.has_option(section, key):
                return self.config.get(section, key)
        except Exception:
            _logger.warning("Unexpected error in ConfigManager.get", exc_info=True)
        return self.defaults.get(section, {}).get(key, fallback)


CFG = ConfigManager()


def _load_ui_font_size(*keys: str, default: int = 11) -> int:
    for key in keys:
        raw = CFG.get('UI', key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return default


def _clamp_font_size(value: int, minimum: int = 8, maximum: int = 28) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, size))

class MergerModel:
    def __init__(self):
        self.data_slots = [None, None, None]
        self.filenames = ["Aucun fichier", "Aucun fichier", "Aucun fichier"]
        self.current_absolute_index = 0
        self.max_index = 0
        self.selected_indices = set() 
        self.total_sn_counts = Counter()
        self.total_de_counts = Counter()
        self.visible_sn_counts = Counter()
        self.visible_de_counts = Counter()
        self.all_known_sn = set()
        self.all_known_de = set()
        self.active_sn_filters = set()
        self.active_de_filters = set()
        self.search_query = "" 
        self.filtered_indices = []
        self.available_sn = set()
        self.available_de = set()
        self._sn_allow_all_when_empty = True
        self._de_allow_all_when_empty = True

    def parse_file(self, file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                return None, f"Encodage : {str(e)}"
        except Exception as e:
            return None, str(e)

        def _normalize_code(prefix: str, num: str, suffix: str) -> str:
            return f"{prefix.upper()}{num}{(suffix or '').lower()}"

        def _extract_codes_from_text(text: str) -> tuple[list[str], list[str]]:
            sn_matches = re.findall(r"\bSN(\d+)([A-Za-z]*)\b", text, re.IGNORECASE)
            de_matches = re.findall(r"\bDE(\d+)([A-Za-z]*)\b", text, re.IGNORECASE)
            sn = {_normalize_code("SN", n, s) for (n, s) in sn_matches}
            de = {_normalize_code("DE", n, s) for (n, s) in de_matches}
            return sorted(sn), sorted(de)

        def _coerce_codes(raw_value: Any) -> List[str]:
            if raw_value is None:
                return []
            tokens: List[str] = []
            if isinstance(raw_value, str):
                tokens = re.split(r"[,\s]+", raw_value)
            elif isinstance(raw_value, (list, tuple, set)):
                for item in raw_value:
                    if isinstance(item, str):
                        tokens.extend(re.split(r"[,\s]+", item))
            codes = set()
            for token in tokens:
                t = token.strip()
                if not t:
                    continue
                match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]*)", t)
                if not match:
                    continue
                prefix, num, suffix = match.groups()
                codes.add(_normalize_code(prefix, num, suffix))
            return sorted(codes)

        def _parse_json_blocks(raw: str) -> Optional[pd.DataFrame]:
            stripped = raw.strip()
            if not stripped:
                return None
            records: Optional[List[Dict[str, Any]]] = None
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                tmp: List[Dict[str, Any]] = []
                for line in stripped.splitlines():
                    candidate = line.strip().rstrip(",")
                    if not candidate:
                        continue
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        tmp = []
                        break
                    if isinstance(obj, dict):
                        tmp.append(obj)
                records = tmp or None
            else:
                if isinstance(parsed, list):
                    records = [obj for obj in parsed if isinstance(obj, dict)]
                elif isinstance(parsed, dict):
                    if any(isinstance(parsed.get(key), str) for key in ("text", "Text")):
                        records = [parsed]
                    elif isinstance(parsed.get("items"), list):
                        records = [obj for obj in parsed["items"] if isinstance(obj, dict)]
            if not records:
                return None
            rows: List[Dict[str, Any]] = []
            for idx, entry in enumerate(records, start=1):
                text_val = entry.get("text") or entry.get("Text")
                if not isinstance(text_val, str):
                    continue
                body = text_val.strip()
                if not body:
                    continue
                header = None
                for key in ("header", "title", "topic", "advice", "name"):
                    candidate = entry.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        header = candidate.strip()
                        break
                if not header:
                    header = f"Text {idx}"
                num_candidate = entry.get("num") or entry.get("Num")
                try:
                    num = int(num_candidate)
                except (TypeError, ValueError):
                    num = idx
                sn_codes = _coerce_codes(entry.get("sn") or entry.get("SN") or entry.get("sn_codes"))
                de_codes = _coerce_codes(entry.get("de") or entry.get("DE") or entry.get("de_codes"))
                auto_sn, auto_de = _extract_codes_from_text(f"{header}\n{body}")
                if not sn_codes:
                    sn_codes = auto_sn
                if not de_codes:
                    de_codes = auto_de
                rows.append(
                    {
                        "Num": num,
                        "Header": header,
                        "Text": body,
                        "sn_codes": sn_codes,
                        "de_codes": de_codes,
                    }
                )
            return pd.DataFrame(rows) if rows else None

        def _parse_xml_blocks(raw: str) -> Optional[pd.DataFrame]:
            stripped = raw.strip()
            if not stripped.startswith("<"):
                return None

            def _load_root(payload: str) -> Optional[ET.Element]:
                try:
                    return ET.fromstring(payload)
                except ET.ParseError:
                    try:
                        return ET.fromstring(f"<root>{payload}</root>")
                    except ET.ParseError:
                        return None

            root = _load_root(stripped)
            if root is None:
                return None
            nodes: List[ET.Element]
            if root.tag.lower() == "text":
                nodes = [root]
            else:
                nodes = list(root.iter("text"))
            if not nodes:
                return None
            rows: List[Dict[str, Any]] = []
            for idx, node in enumerate(nodes, start=1):
                body = "".join(node.itertext()).strip()
                if not body:
                    continue
                header = None
                for key in ("header", "title", "topic", "advice", "name"):
                    attr_val = node.get(key)
                    if attr_val and attr_val.strip():
                        header = attr_val.strip()
                        break
                if not header:
                    header = f"Text {idx}"
                num_attr = node.get("num") or node.get("id")
                try:
                    num = int(num_attr)
                except (TypeError, ValueError):
                    num = idx
                sn_codes = _coerce_codes(node.get("sn") or node.get("sn_codes"))
                de_codes = _coerce_codes(node.get("de") or node.get("de_codes"))
                auto_sn, auto_de = _extract_codes_from_text(f"{header}\n{body}")
                if not sn_codes:
                    sn_codes = auto_sn
                if not de_codes:
                    de_codes = auto_de
                rows.append(
                    {
                        "Num": num,
                        "Header": header,
                        "Text": body,
                        "sn_codes": sn_codes,
                        "de_codes": de_codes,
                    }
                )
            return pd.DataFrame(rows) if rows else None

        json_df = _parse_json_blocks(content)
        if json_df is not None and not json_df.empty:
            return json_df, None

        xml_df = _parse_xml_blocks(content)
        if xml_df is not None and not xml_df.empty:
            return xml_df, None

        rows: List[Dict[str, Any]] = []
        sec_re = re.compile(r"\\sub(?:sub)?section\*?\{([^}]*)\}", re.IGNORECASE)
        matches = list(sec_re.finditer(content))
        if matches:
            for idx, match in enumerate(matches, start=1):
                header = match.group(1).strip()
                start = match.end()
                end = matches[idx].start() if idx < len(matches) else len(content)
                body = content[start:end].strip()
                sn_codes, de_codes = _extract_codes_from_text(f"{header}\n{body}")
                rows.append(
                    {
                        "Num": idx,
                        "Header": header,
                        "Text": body,
                        "sn_codes": sn_codes,
                        "de_codes": de_codes,
                    }
                )
            return pd.DataFrame(rows), None

        txt_re = re.compile(
            r"(?:^|\n)\s*Text\s+(\d+)\s*[–-]\s*(.*?)(?=\n\s*Text\s+\d+\s*[–-]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        chunks = list(txt_re.finditer(content))
        if chunks:
            for idx, chunk in enumerate(chunks, start=1):
                block = chunk.group(0).strip()
                lines = block.splitlines()
                header = lines[0].strip() if lines else f"Text {idx}"
                body = "\n".join(lines[1:]).strip()
                sn_codes, de_codes = _extract_codes_from_text(f"{header}\n{body}")
                rows.append(
                    {
                        "Num": idx,
                        "Header": header,
                        "Text": body,
                        "sn_codes": sn_codes,
                        "de_codes": de_codes,
                    }
                )
            return pd.DataFrame(rows), None

        base = os.path.basename(file_path)
        sn_codes, de_codes = _extract_codes_from_text(content)
        return (
            pd.DataFrame(
                [
                    {
                        "Num": 1,
                        "Header": base,
                        "Text": content.strip(),
                        "sn_codes": sn_codes,
                        "de_codes": de_codes,
                    }
                ]
            ),
            None,
        )

    def _recalculate_globals(self):
        self.total_sn_counts = Counter()
        self.total_de_counts = Counter()
        self.all_known_sn = set()
        self.all_known_de = set()
        for df in self.data_slots:
            if df is not None:
                for _, row in df.iterrows():
                    self.total_sn_counts.update(row['sn_codes'])
                    self.total_de_counts.update(row['de_codes'])
                    self.all_known_sn.update(row['sn_codes'])
                    self.all_known_de.update(row['de_codes'])
        self.active_sn_filters = self.all_known_sn.copy()
        self.active_de_filters = self.all_known_de.copy()
        self._sn_allow_all_when_empty = True
        self._de_allow_all_when_empty = True

    def _calculate_available_codes(self):
        indices_sn_ok = []
        if not self.active_sn_filters:
            if self._sn_allow_all_when_empty:
                indices_sn_ok = [idx for idx in range(self.max_index) if self._match_search(idx)]
            else:
                indices_sn_ok = []
        else:
            for idx in range(self.max_index):
                if not self._match_search(idx): continue
                idx_sn = set()
                for df in self.data_slots:
                    if df is not None and idx < len(df): idx_sn.update(df.iloc[idx]['sn_codes'])
                if not self.active_sn_filters.isdisjoint(idx_sn): indices_sn_ok.append(idx)

        self.available_de = set()
        for idx in indices_sn_ok:
            for df in self.data_slots:
                if df is not None and idx < len(df): self.available_de.update(df.iloc[idx]['de_codes'])

        indices_de_ok = []
        if not self.active_de_filters:
            if self._de_allow_all_when_empty:
                indices_de_ok = [idx for idx in range(self.max_index) if self._match_search(idx)]
            else:
                indices_de_ok = []
        else:
            for idx in range(self.max_index):
                if not self._match_search(idx): continue
                idx_de = set()
                for df in self.data_slots:
                    if df is not None and idx < len(df): idx_de.update(df.iloc[idx]['de_codes'])
                if not self.active_de_filters.isdisjoint(idx_de): indices_de_ok.append(idx)

        self.available_sn = set()
        for idx in indices_de_ok:
            for df in self.data_slots:
                if df is not None and idx < len(df): self.available_sn.update(df.iloc[idx]['sn_codes'])

    def _match_search(self, idx):
        if not self.search_query: return True
        q = self.search_query.lower()
        for df in self.data_slots:
            if df is not None and idx < len(df):
                row = df.iloc[idx]
                if q in row['Header'].lower() or q in row['Text'].lower(): return True
        return False

    def _apply_filters(self):
        self._calculate_available_codes()
        if self.max_index == 0:
            self.filtered_indices = []
            self.visible_sn_counts = Counter()
            self.visible_de_counts = Counter()
            return
        valid = []
        for idx in range(self.max_index):
            if not self._match_search(idx): continue
            idx_sn = set()
            idx_de = set()
            for df in self.data_slots:
                if df is not None and idx < len(df):
                    idx_sn.update(df.iloc[idx]['sn_codes'])
                    idx_de.update(df.iloc[idx]['de_codes'])
            if not self.active_sn_filters:
                match_sn = self._sn_allow_all_when_empty
            else:
                match_sn = not self.active_sn_filters.isdisjoint(idx_sn)
            if not self.active_de_filters:
                match_de = self._de_allow_all_when_empty
            else:
                match_de = not self.active_de_filters.isdisjoint(idx_de)
            if match_sn and match_de: valid.append(idx)
        self.filtered_indices = valid
        self._update_visible_counts()
        if self.filtered_indices:
            if self.current_absolute_index not in self.filtered_indices:
                self.current_absolute_index = self.filtered_indices[0]

    def _update_visible_counts(self):
        self.visible_sn_counts = Counter()
        self.visible_de_counts = Counter()
        if not self.filtered_indices:
            return
        for idx in self.filtered_indices:
            sn_codes = set()
            de_codes = set()
            for df in self.data_slots:
                if df is not None and idx < len(df):
                    row = df.iloc[idx]
                    sn_codes.update(row['sn_codes'])
                    de_codes.update(row['de_codes'])
            self.visible_sn_counts.update(sn_codes)
            self.visible_de_counts.update(de_codes)

    def set_search_query(self, query): self.search_query = query.strip(); self._apply_filters()
    def toggle_filter_sn(self, code):
        if code in self.active_sn_filters:
            self.active_sn_filters.remove(code)
        else:
            self.active_sn_filters.add(code)
        if not self.active_sn_filters:
            self._sn_allow_all_when_empty = False
        self._apply_filters()
    def toggle_filter_de(self, code):
        if code in self.active_de_filters:
            self.active_de_filters.remove(code)
        else:
            self.active_de_filters.add(code)
        if not self.active_de_filters:
            self._de_allow_all_when_empty = False
        self._apply_filters()
    def select_all_sn(self):
        self.active_sn_filters = self.all_known_sn.copy()
        self._sn_allow_all_when_empty = True
        self._apply_filters()
    def select_none_sn(self):
        self.active_sn_filters.clear()
        self._sn_allow_all_when_empty = False
        self._apply_filters()
    def select_all_de(self):
        self.active_de_filters = self.all_known_de.copy()
        self._de_allow_all_when_empty = True
        self._apply_filters()
    def select_none_de(self):
        self.active_de_filters.clear()
        self._de_allow_all_when_empty = False
        self._apply_filters()

    def load_file_into_slot(self, slot_index, file_path):
        new_df, error = self.parse_file(file_path)
        if error: return False, f"Erreur: {error}"
        if new_df.empty: return False, "Vide."
        ref = next((df for df in self.data_slots if df is not None), None)
        if ref is not None and len(new_df) != len(ref): return False, "Désalignement."
        self.data_slots[slot_index] = new_df
        self.filenames[slot_index] = os.path.basename(file_path)
        if self.max_index == 0: self.max_index = len(new_df)
        self._recalculate_globals()
        self._apply_filters()
        return True, "OK"

    def clear_slot(self, slot_index):
        self.data_slots[slot_index] = None
        self.filenames[slot_index] = "Aucun fichier"
        if all(df is None for df in self.data_slots):
            self.max_index = 0
            self.current_absolute_index = 0
            self.selected_indices.clear()
            self.filtered_indices = []
            self._recalculate_globals()
        else:
            self._recalculate_globals()
            self._apply_filters()

    def toggle_selection_current(self):
        if not self.filtered_indices: return
        idx = self.current_absolute_index
        if idx in self.selected_indices: self.selected_indices.remove(idx)
        else: self.selected_indices.add(idx)

    def set_absolute_index(self, index):
        if 0 <= index < self.max_index:
            self.current_absolute_index = index
            return True
        return False

    def reset_all_selections(self): self.selected_indices.clear()

    def backup_selection(self, filename="backup_before_reset.json"):
        if not self.selected_indices: return
        ref_df = next((df for df in self.data_slots if df is not None), None)
        if ref_df is not None:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    for idx in self.selected_indices: f.write(f"{ref_df.iloc[idx]['Header']}\n")
                return True
            except (OSError, UnicodeError, KeyError, IndexError):
                _logger.warning("[MERGER] backup_selection failed", exc_info=True)
                return False
        return False


    def import_selection(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f: headers = [l.strip() for l in f if l.strip()]
            ref_df = next((df for df in self.data_slots if df is not None), None)
            if ref_df is None: return False, "Chargez un corpus d'abord."
            header_map = {r['Header'].strip(): i for i, r in ref_df.iterrows()}
            new_sel = set()
            cnt = 0
            for h in headers:
                if h in header_map: new_sel.add(header_map[h]); cnt += 1
            self.selected_indices = new_sel
            return True, f"{cnt} sélectionnés."
        except Exception as e: return False, str(e)

    def get_view_data(self, slot_index):
        df = self.data_slots[slot_index]
        if df is None: return None
        if not self.filtered_indices: return None
        if self.current_absolute_index >= len(df): return None
        row = df.iloc[self.current_absolute_index]
        return {
            "num": row['Num'], "header": row['Header'], "text": row['Text'],
            "filename": self.filenames[slot_index],
            "is_selected": (self.current_absolute_index in self.selected_indices)
        }

    def get_detailed_stats(self):
        visible_set = set(self.filtered_indices or [])
        selected_visible = [idx for idx in self.selected_indices if idx in visible_set]

        sel_sn = Counter()
        sel_de = Counter()
        for idx in self.selected_indices:
            sn_codes = set()
            de_codes = set()
            for df in self.data_slots:
                if df is not None and idx < len(df):
                    sn_codes.update(df.iloc[idx]['sn_codes'])
                    de_codes.update(df.iloc[idx]['de_codes'])
            sel_sn.update(sn_codes)
            sel_de.update(de_codes)

        sel_visible_sn = Counter()
        sel_visible_de = Counter()
        for idx in selected_visible:
            sn_codes = set()
            de_codes = set()
            for df in self.data_slots:
                if df is not None and idx < len(df):
                    sn_codes.update(df.iloc[idx]['sn_codes'])
                    de_codes.update(df.iloc[idx]['de_codes'])
            sel_visible_sn.update(sn_codes)
            sel_visible_de.update(de_codes)

        current_sn = set()
        current_de = set()
        if 0 <= self.current_absolute_index < self.max_index:
            for df in self.data_slots:
                if df is not None and self.current_absolute_index < len(df):
                    row = df.iloc[self.current_absolute_index]
                    current_sn.update(row['sn_codes'])
                    current_de.update(row['de_codes'])

        return {
            "count_selected": len(self.selected_indices),
            "count_selected_visible": len(selected_visible),
            "count_filtered": len(self.filtered_indices),
            "total_texts": self.max_index,
            "sel_sn": sel_sn,
            "sel_de": sel_de,
            "sel_visible_sn": sel_visible_sn,
            "sel_visible_de": sel_visible_de,
            "total_sn": self.total_sn_counts,
            "total_de": self.total_de_counts,
            "visible_sn": self.visible_sn_counts,
            "visible_de": self.visible_de_counts,
            "filters_sn": self.active_sn_filters,
            "filters_de": self.active_de_filters,
            "available_sn": self.available_sn,
            "available_de": self.available_de,
            "current_sn_codes": current_sn,
            "current_de_codes": current_de,
            "current_selected": (self.current_absolute_index in self.selected_indices)
        }

    def get_nav_list_data(self):
        if not self.filtered_indices: return []
        ref = next((df for df in self.data_slots if df is not None), None)
        if ref is None: return []
        res = []
        for idx in self.filtered_indices:
            row = ref.iloc[idx]
            sn_codes = ",".join(row['sn_codes']) if row['sn_codes'] else "-"
            de_codes = ",".join(row['de_codes']) if row['de_codes'] else "-"
            codes = f" — SN:{sn_codes} | DE:{de_codes}"
            res.append((idx, row['Header'], idx in self.selected_indices, codes))
        return res

    def get_text_content(self, slot_index):
        df = self.data_slots[slot_index]
        if df is None: return None, None
        row = df.iloc[self.current_absolute_index]
        return f"\\subsubsection{{{row['Header']}}}\n{row['Text']}", f"{os.path.splitext(self.filenames[slot_index])[0]}_{row['Num']}.txt"

    def get_all_views_content(self):
        if not self.filtered_indices: return None
        txt = ""
        for i, df in enumerate(self.data_slots):
            if df is not None:
                txt += f"--- VUE {i+1} ---\n{df.iloc[self.current_absolute_index]['Text']}\n\n"
        return txt if txt else None

    def get_batch_export_data(self):
        if not self.selected_indices: return None
        sorted_indices = sorted(list(self.selected_indices))
        data = {'slots': {}}
        ref = next((df for df in self.data_slots if df is not None), None)
        if ref is not None:
            data['headers'] = [ref.iloc[idx]['Header'] for idx in sorted_indices]
        for i, df in enumerate(self.data_slots):
            if df is not None:
                content = []
                for idx in sorted_indices:
                    r = df.iloc[idx]
                    content.append(f"\\subsubsection{{{r['Header']}}}\n{r['Text']}")
                data['slots'][i] = {'filename': self.filenames[i], 'content': "\n\n".join(content)}
        return data
    
    def move_index(self, direction):
        if not self.filtered_indices: return False
        try: current_pos = self.filtered_indices.index(self.current_absolute_index)
        except ValueError: current_pos = -1
        new_pos = current_pos + direction
        if 0 <= new_pos < len(self.filtered_indices):
            self.current_absolute_index = self.filtered_indices[new_pos]
            return True
        return False


class GeneratorTab(tk.Frame):
    def __init__(self, master):
        super().__init__(master, padx=10, pady=10)
        self.base_title = "Narremgen - simple gui "

        self.topic_var = tk.StringVar(value="Protecting the seas and oceans and lakes")
        self.selected_variant = tk.StringVar(value="direct")
        self.log_queue = queue.Queue()
        self._log_line_limit = 2000
        self.active_workdir_var = tk.StringVar(value="Workdir: -")
        self._logs_visible = tk.BooleanVar(value=False)
        self._current_cancel_event: Optional[threading.Event] = None
        self._current_worker_thread: Optional[threading.Thread] = None
        self._stop_requested = False

        self._initial_env = dict(os.environ)   # snapshot “au boot”
        self._injected_env_keys: set[str] = set()

        self.packaged_assets_dir = DEFAULT_ASSETS_DIR
        self.assets_override: Optional[Path] = None
        self.output_override: Optional[Path] = None
        self.assets_dir = self.packaged_assets_dir
        self.output_dir = DEFAULT_OUTPUT_DIR

        self.neutral_n_batches = DEFAULT_NEUTRAL_N_BATCHES
        self.neutral_n_per_batch = DEFAULT_NEUTRAL_N_PER_BATCH
        self.variant_batch_size = DEFAULT_VARIANT_BATCH_SIZE
        self.simple_max_tokens = DEFAULT_SIMPLE_MAX_TOKENS
        self.variants_overwrite_existing = True 

        self.themes_min = DEFAULT_THEMES_MIN
        self.themes_max = DEFAULT_THEMES_MAX
        self.themes_batch_size = DEFAULT_THEMES_BATCH_SIZE

        self.llm_models = DEFAULT_LLM_MODELS.copy()
        self.variants_registry = {}

        self._threads = []
        self.is_running = False
        self._controlled_widgets = []
        self.gui_log_level_name = "INFO"
        self._widget_state_cache: Dict[tk.Widget, str] = {}
        self._completion_queue: queue.Queue = queue.Queue()
        self._pipeline_file_handler: Optional[logging.Handler] = None
        self._poll_after_id: Optional[str] = None
        self._polling_active = True
        self._disabled_widget_exclusions: Set[tk.Widget] = set()
        self._llm_preflight_cache: Dict[str, Any] = {
            "fingerprint": None,
            "timestamp": 0.0,
            "status": False,
            "results": {},
            "auth_map": {},
            "scope": None,
        }

        self._diagnostics_queue: queue.Queue = queue.Queue()
        self._diagnostics_worker_running = False
        self._diagnostics_attention_flag = False
        self._diagnostics_banner_var = tk.StringVar(value="Diagnostics not run yet.")
        self._diagnostics_banner_level = "info"
        self._diagnostics_alert_var = tk.StringVar(value="")
        self._diagnostics_hide_unused_var = tk.BooleanVar(value=False)
        self._diagnostics_session_visible = False
        self._diagnostics_env_rows: list[tuple[str, str, str, str]] = []
        self._diagnostics_missing_rows: list[tuple[str, str]] = []
        self._diagnostics_auth_map: dict[str, dict[str, Any]] = {}
        self._diagnostics_last_scope = "default"
        self._session_selected_key = tk.StringVar(value='')
        self._session_entry_var = tk.StringVar(value='')
        self._session_entry_status = tk.StringVar(value='none')

        self._load_config()
        self._load_variants_config()

        self.llm = None

        self._log_handler = self._install_gui_logging_handler()
        self._build_widgets()
        self.bind_all("<Control-l>", self._toggle_logs_shortcut)
        self._update_log_visibility()
        self._set_running_ui_state(False)
        self._poll_log_queue()

 
    def _load_config(self):
        cfg = configparser.ConfigParser()

        cfg["paths"] = {
            "assets_override": "",
            "output_override": "",
        }

        cfg["pipeline"] = {
            "neutral_n_batches": str(DEFAULT_NEUTRAL_N_BATCHES),
            "neutral_n_per_batch": str(DEFAULT_NEUTRAL_N_PER_BATCH),
            "variant_batch_size": str(DEFAULT_VARIANT_BATCH_SIZE),
            "simple_max_tokens": str(DEFAULT_SIMPLE_MAX_TOKENS),
        }
        cfg["variants"] = {
            "overwrite_existing": "true",
        }
        cfg["themes"] = {
            "themes_n_themes_min": str(DEFAULT_THEMES_MIN),
            "themes_n_themes_max": str(DEFAULT_THEMES_MAX),
            "themes_batch_size": str(DEFAULT_THEMES_BATCH_SIZE),
        }
        cfg["llm"] = DEFAULT_LLM_MODELS.copy()
        cfg["logging"] = {
            "gui_level": "INFO",
            "file_level": "DEBUG",
        }

        if CONFIG_PATH.exists():
            try:
                cfg.read(CONFIG_PATH, encoding="utf-8")
            except Exception:
                self.log_queue.put("Unexpected error in GeneratorTab._load_config")

        def _get_int(section, key, default):
            try:
                return cfg.getint(section, key, fallback=default)
            except ValueError:
                return default

        def _clean(value: Optional[str]) -> str:
            return (value or "").strip()

        def _expand_path(raw: str) -> Optional[Path]:
            raw = _clean(raw)
            if not raw:
                return None
            try:
                return Path(raw).expanduser()
            except Exception as exc:
                self.log_queue.put(f"[CONFIG] Invalid path '{raw}': {exc}")
                return None

        def _resolve_existing_dir(raw: str) -> Optional[Path]:
            candidate = _expand_path(raw)
            if not candidate:
                return None
            if candidate.exists():
                return candidate
            self.log_queue.put(f"[CONFIG] Ignoring missing assets override: {candidate}")
            return None

        def _is_subpath(path: Path, root: Path) -> bool:
            try:
                path.resolve().relative_to(root.resolve())
                return True
            except ValueError:
                return False

        def _validate_legacy_output(candidate: Optional[Path], source_key: str) -> Optional[Path]:
            if not candidate:
                return None
            try:
                resolved = candidate.expanduser().resolve()
            except Exception as exc:
                self.log_queue.put(f"[CONFIG] Ignoring legacy {source_key}: invalid path ({candidate}): {exc}")
                return None
            if not resolved.exists():
                self.log_queue.put(f"[CONFIG] Ignoring legacy {source_key}: path missing ({resolved})")
                return None
            if not os.access(resolved, os.W_OK):
                self.log_queue.put(f"[CONFIG] Ignoring legacy {source_key}: not writable ({resolved})")
                return None
            home = Path.home()
            if not _is_subpath(resolved, home):
                self.log_queue.put(f"[CONFIG] Ignoring legacy {source_key}: outside user home ({resolved})")
                return None
            return resolved

        self.assets_override = _resolve_existing_dir(cfg.get("paths", "assets_override", fallback=""))
        if not self.assets_override:
            for legacy_key in ("user_assets_dir", "assets_dir"):
                legacy = cfg.get("paths", legacy_key, fallback="")
                if legacy and legacy.lower().strip() in ("auto", "package", "default"):
                    continue
                legacy_path = _resolve_existing_dir(legacy)
                if legacy_path:
                    self.assets_override = legacy_path
                    self.log_queue.put(f"[CONFIG] Migrated legacy {legacy_key} -> assets override ({legacy_path})")
                    break

        self.output_override = None
        output_candidate = _expand_path(cfg.get("paths", "output_override", fallback=""))
        if output_candidate:
            self.output_override = output_candidate
        else:
            legacy_output = _expand_path(cfg.get("paths", "output_dir", fallback=""))
            validated = _validate_legacy_output(legacy_output, "output_dir")
            if validated and validated != DEFAULT_OUTPUT_DIR:
                self.output_override = validated
                self.log_queue.put(f"[CONFIG] Migrated legacy output_dir -> output override ({validated})")

        self._refresh_effective_paths()
        self.api_key_file = cfg.get("paths", "api_key_file", fallback="").strip()
                
        self.neutral_n_batches = _get_int("pipeline", "neutral_n_batches", DEFAULT_NEUTRAL_N_BATCHES)
        self.neutral_n_per_batch = _get_int("pipeline", "neutral_n_per_batch", DEFAULT_NEUTRAL_N_PER_BATCH)
        self.variant_batch_size = _get_int("pipeline", "variant_batch_size", DEFAULT_VARIANT_BATCH_SIZE)
        self.simple_max_tokens = _get_int("pipeline", "simple_max_tokens", DEFAULT_SIMPLE_MAX_TOKENS)

        self.variants_overwrite_existing = cfg.getboolean("variants", "overwrite_existing", fallback=True)

        self.themes_min = _get_int("themes", "themes_n_themes_min", DEFAULT_THEMES_MIN)
        self.themes_max = _get_int("themes", "themes_n_themes_max", DEFAULT_THEMES_MAX)
        self.themes_batch_size = _get_int("themes", "themes_batch_size", DEFAULT_THEMES_BATCH_SIZE)

        self.llm_models = {}
        for key, default_model in DEFAULT_LLM_MODELS.items():
            self.llm_models[key] = cfg.get("llm", key, fallback=default_model)

        gui_level_name = cfg.get("logging", "gui_level", fallback="INFO").upper()
        if gui_level_name not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            gui_level_name = "INFO"
        self.gui_log_level_name = gui_level_name

    def _save_config(self):
        cfg = configparser.ConfigParser()
        cfg["paths"] = {
            "assets_override": str(self.assets_override or ""),
            "output_override": str(self.output_override or ""),
            "api_key_file": str(getattr(self, "api_key_file", "") or ""),
        }
        cfg["pipeline"] = {
            "neutral_n_batches": str(self.neutral_n_batches),
            "neutral_n_per_batch": str(self.neutral_n_per_batch),
            "variant_batch_size": str(self.variant_batch_size),
            "simple_max_tokens": str(self.simple_max_tokens),
        }
        cfg["variants"] = {
            "overwrite_existing": str(bool(getattr(self, "variants_overwrite_existing", True))),
        }
        cfg["themes"] = {
            "themes_n_themes_min": str(self.themes_min),
            "themes_n_themes_max": str(self.themes_max),
            "themes_batch_size": str(self.themes_batch_size),
        }
        cfg["llm"] = self.llm_models.copy()
        cfg["logging"] = {
            "gui_level": self.gui_log_level_name,
            "file_level": "DEBUG",
        }

        try:
            with _interprocess_lock(LOCK_PATH):
                _atomic_write_ini(CONFIG_PATH, cfg.write)
            self.log_queue.put(f"[CONFIG] Saved config to {CONFIG_PATH}")
        except Exception as exc:
            self.log_queue.put(f"[CONFIG] ERROR saving {CONFIG_PATH}: {exc}")
            self.log_queue.put(traceback.format_exc())

    def _refresh_effective_paths(self):
        repo_settings_dir = Path(__file__).resolve().parent / "settings"
        assets_candidates: list[tuple[str, Optional[Path]]] = []
        if self.assets_override:
            assets_candidates.append(("assets override", self.assets_override))
        assets_candidates.append(("packaged assets", self.packaged_assets_dir))
        if repo_settings_dir not in [p for _, p in assets_candidates]:
            assets_candidates.append(("local settings", repo_settings_dir))

        selected_assets: Optional[Path] = None
        for label, candidate in assets_candidates:
            if candidate and candidate.exists():
                selected_assets = candidate
                if label != "assets override" and (not self.assets_override or candidate != self.assets_override):
                    self.log_queue.put(f"[CONFIG] Using {label}: {candidate}")
                break
            if candidate:
                self.log_queue.put(f"[CONFIG] {label} missing: {candidate}")
        if not selected_assets:
            selected_assets = self.packaged_assets_dir
        self.assets_dir = selected_assets

        output_candidate = self.output_override or DEFAULT_OUTPUT_DIR
        try:
            output_candidate.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.log_queue.put(f"[CONFIG] Cannot create output dir {output_candidate}: {exc}")
        self.output_dir = output_candidate

        if hasattr(self, "assets_effective_var"):
            self.assets_effective_var.set(str(self.assets_dir))
        if hasattr(self, "output_effective_var"):
            self.output_effective_var.set(str(self.output_dir))
        if hasattr(self, "output_dir_var"):
            self.output_dir_var.set(str(self.output_dir))


    def _load_variants_config(self):
        cfg = configparser.ConfigParser()
        if VARIANTS_CONFIG_PATH.exists():
            try:
                cfg.read(VARIANTS_CONFIG_PATH, encoding="utf-8")
            except Exception:
                self.log_queue.put("Unexpected error in GeneratorTab._load_variants_config")

        variants = {}
        for section in cfg.sections():
            if not section.startswith("variant."):
                continue
            name = section.split(".", 1)[1].strip()
            if not name:
                continue
            variants[name] = {
                "name": name,
                "title": cfg.get(section, "title", fallback=name),
                "prompt_path": cfg.get(section, "prompt_path", fallback=""),
                "use_sn": cfg.getboolean(section, "use_sn", fallback=False),
                "use_de": cfg.getboolean(section, "use_de", fallback=False),
                "use_k": cfg.getboolean(section, "use_k", fallback=False),
                "use_ops": cfg.getboolean(section, "use_ops", fallback=False),
                "active": cfg.getboolean(section, "active", fallback=True),
                "kind": cfg.get(section, "kind", fallback="llm"),
            }

        if not variants:
            variants = {
                "direct": {
                    "name": "direct",
                    "title": "Direct",
                    "prompt_path": "simple_prompt_variants.txt",
                    "use_sn": True,
                    "use_de": True,
                    "use_k": True,
                    "use_ops": False,
                    "active": True,
                    "kind": "llm",
                },
                "formal": {
                    "name": "formal",
                    "title": "Formal",
                    "prompt_path": "simple_prompt_variants_formal.txt",
                    "use_sn": False,
                    "use_de": False,
                    "use_k": False,
                    "use_ops": False,
                    "active": True,
                    "kind": "llm",
                },
            }

        self.variants_registry = variants
        if self.selected_variant.get() not in self.variants_registry:
            for name, data in self.variants_registry.items():
                if data.get("active", True):
                    self.selected_variant.set(name)
                    break
            else:
                self.selected_variant.set(next(iter(self.variants_registry)))

    def _save_variants_config(self):
        cfg = configparser.ConfigParser()
        for name, data in self.variants_registry.items():
            section = f"variant.{name}"
            cfg[section] = {
                "title": data.get("title", name),
                "prompt_path": data.get("prompt_path", ""),
                "use_sn": str(bool(data.get("use_sn", False))),
                "use_de": str(bool(data.get("use_de", False))),
                "use_k": str(bool(data.get("use_k", False))),
                "use_ops": str(bool(data.get("use_ops", False))),
                "active": str(bool(data.get("active", True))),
                "kind": data.get("kind", "llm"),
            }

        try:
            with _interprocess_lock(LOCK_PATH):
                _atomic_write_ini(VARIANTS_CONFIG_PATH, cfg.write)
            self.log_queue.put(f"[CONFIG] Saved variants to {VARIANTS_CONFIG_PATH}")
        except Exception as exc:
            self.log_queue.put(f"[CONFIG] ERROR saving {VARIANTS_CONFIG_PATH}: {exc}")
            self.log_queue.put(traceback.format_exc())

    @staticmethod
    def _format_models_map(d: dict) -> str:
        items = sorted((str(k), str(v)) for k, v in (d or {}).items())
        return "\n".join([f"  - {k}: {v}" for k, v in items]) or "  (empty)"

    def ensure_llm_ready_or_route_to_diagnostics(self) -> bool:
        fingerprint = self._llm_env_fingerprint()
        if self._is_preflight_cache_valid(fingerprint):
            self._set_diagnostics_banner('Diagnostics cache OK (<=10 min).', 'ok')
            self._set_diagnostics_attention(False)
            return True

        self._refresh_env_snapshot()
        missing = list(self._diagnostics_missing_rows)
        if missing:
            missing_desc = ', '.join(f"{prov}:{env}" for prov, env in missing)
            self.log_queue.put(f"[LLM] Missing env var(s): {missing_desc}")
            self._set_diagnostics_banner('Missing env vars. Inject keys in the Diagnostics tab.', 'warn')
        else:
            self._set_diagnostics_banner('LLM diagnostics required before launching pipelines.', 'warn')

        self._set_diagnostics_attention(True)
        self._set_diagnostics_alert('LLM diagnostics pending. Open the Diagnostics tab.', severity='warn')
        self._focus_diagnostics_tab()
        if not missing and not self._diagnostics_worker_running:
            self._start_diagnostics_test(scope='default', auto=True)
        return False

    def _refresh_env_snapshot(self) -> None:
        rows, missing = self._env_status_snapshot()
        self._diagnostics_env_rows = rows
        self._diagnostics_missing_rows = missing
        self._refresh_diagnostics_env_tree()
        if self._diagnostics_session_visible:
            self._refresh_session_entries()

    def _env_status_snapshot(self) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
        rows: list[tuple[str, str, str, str]] = []
        missing: list[tuple[str, str]] = []
        configured = self._providers_from_models()
        providers = set(configured)

        for provider, env_keys in KNOWN_PROVIDER_ENV_VARS.items():
            if provider in providers:
                continue
            if any(os.environ.get(env) for env in env_keys):
                providers.add(provider)

        for provider in sorted(providers):
            groups = _required_env_for_provider(provider)
            if not groups:
                fallbacks = KNOWN_PROVIDER_ENV_VARS.get(provider, [])
                if not fallbacks:
                    continue
                groups = [[env] for env in fallbacks]

            for group in groups:
                label = " / ".join(group)
                has_value = any((os.environ.get(env_key) or "").strip() for env_key in group)
                status = "present" if has_value else "missing"
                usage = "used" if provider in configured else "not used"
                rows.append((provider or "-", label, status, usage))
                if not has_value:
                    missing.append((provider or "-", label))

        rows.sort(key=lambda item: (item[0], item[1]))
        return rows, missing

    def _providers_from_models(self) -> set[str]:
        providers: set[str] = set()
        for spec in (self.llm_models or {}).values():
            provider, _ = _split_provider(spec or "")
            if provider:
                providers.add(provider)
        return providers

    def _restore_initial_env(self) -> None:
        snapshot = getattr(self, "_initial_env", None) or {}
        target_keys: set[str] = set(self._injected_env_keys)
        for env_keys in KNOWN_PROVIDER_ENV_VARS.values():
            for env_key in env_keys:
                if env_key:
                    target_keys.add(env_key)

        for key in target_keys:
            if key in snapshot:
                os.environ[key] = snapshot[key]
            else:
                os.environ.pop(key, None)

        self._injected_env_keys.clear()

    def _llm_env_fingerprint(self) -> str:
        relevant_keys: set[str] = set()
        for provider in self._providers_from_models():
            for group in _required_env_for_provider(provider):
                for env_key in group:
                    if env_key:
                        relevant_keys.add(env_key)
        for env_keys in KNOWN_PROVIDER_ENV_VARS.values():
            for env_key in env_keys:
                if env_key:
                    relevant_keys.add(env_key)

        payload = {
            "models": self.llm_models or {},
            "api_key_file": getattr(self, "api_key_file", "") or "",
            "env": {key: os.environ.get(key, "") for key in sorted(relevant_keys)},
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _is_preflight_cache_valid(self, fingerprint: Optional[str], ttl_seconds: float = 600.0) -> bool:
        cache = self._llm_preflight_cache or {}
        if not fingerprint:
            return False
        cached_fp = cache.get("fingerprint")
        if not cache.get("status") or not cached_fp or cached_fp != fingerprint:
            return False
        ts = cache.get("timestamp") or 0.0
        if ts <= 0:
            return False
        return (time.time() - ts) <= ttl_seconds

    def _start_diagnostics_test(self, scope: str, provider_models: Optional[list[str]] = None, auto: bool = False) -> None:
        if self._diagnostics_worker_running:
            self.log_queue.put('[LLM] Diagnostics already running.')
            return
        self._refresh_env_snapshot()
        if self._diagnostics_missing_rows:
            self._set_diagnostics_banner('Missing env vars. Update keys first.', 'warn')
            return
        #models = list(provider_models or [])
        try:
            for k, var in getattr(self, "llm_vars", {}).items():
                v = (var.get() or "").strip()
                if v:
                    self.llm_models[k] = v
        except Exception:
            pass
        models = list(provider_models or [])

        if not models:
            if scope == 'all':
                models = list(dict.fromkeys(filter(None, (self.llm_models or {}).values())))
            else:
                fallback = next(iter((self.llm_models or {}).values()), 'openai\\gpt-4o-mini')
                models = [self.llm_models.get('NARRATIVE', fallback)]
        models = [m for m in models if m]
        if not models:
            self._set_diagnostics_banner('No LLM models configured. Update Settings > LLM models.', 'warn')
            return
        fingerprint = self._llm_env_fingerprint()
        self._diagnostics_worker_running = True
        self._diagnostics_last_scope = scope
        if not auto:
            self._set_diagnostics_banner('Running LLM dry-test...', 'info')
        self.log_queue.put(f"[LLM] Running diagnostics ({scope}, {len(models)} model(s)).")
        worker = threading.Thread(
            target=self._llm_diagnostics_worker,
            args=(models, fingerprint, scope),
            daemon=True,
            name='llm-diagnostics',
        )
        worker.start()

    def _llm_diagnostics_worker(self, provider_models: list[str], fingerprint: str, scope: str) -> None:
        payload: dict[str, Any] = {'state': 'error', 'message': 'Unknown diagnostics error.', 'scope': scope}
        try:
            default_model = self.llm_models.get('NARRATIVE', 'openai\\gpt-4o-mini')
            LLMConnect.init_global(
                llmodels=self.llm_models,
                default_model=default_model,
                temperature=float(getattr(self, 'temperature', 0.7)),
                max_tokens=int(getattr(self, 'max_tokens', 1024)),
                request_timeout=int(getattr(self, 'request_timeout', 600)),
            )
            self.llm = LLMConnect.get_global()
            dry_results = self.llm.dry_test_models(provider_models, verbose=True)
            results_dicts = [asdict(res) for res in dry_results]
            #overall_ok = all(res.ok for res in dry_results)
            overall_ok = all(
                                (res.ok is True) or str(res.message).startswith("SKIPPED")
                                for res in dry_results
                            )
            payload = {
                'state': 'ok' if overall_ok else 'failed',
                'results': results_dicts,
                'fingerprint': fingerprint if overall_ok else None,
                'scope': scope,
            }
        except Exception as exc:
            payload = {
                'state': 'error',
                'message': self._scrub_secret(str(exc)),
                'scope': scope,
            }
        finally:
            self._diagnostics_queue.put(payload)

    def _handle_diagnostics_payload(self, payload: dict[str, Any]) -> None:
        self._diagnostics_worker_running = False
        state = payload.get('state')
        if state in {'ok', 'failed'}:
            results = payload.get('results') or []
            fingerprint = payload.get('fingerprint')
            ok = state == 'ok'
            self._apply_diagnostics_results(results, fingerprint, ok)
            if ok:
                self._set_diagnostics_banner('Diagnostics OK. Pipelines are unlocked.', 'ok')
                self._set_diagnostics_attention(False)
                self._set_diagnostics_alert(None)
            else:
                self._set_diagnostics_banner('Dry-test failed. Review AuthStatus table below.', 'error')
                self._set_diagnostics_attention(True)
                self._set_diagnostics_alert('LLM diagnostics failed. Open Diagnostics for details.', severity='error')
            return
        message = payload.get('message', 'Unknown diagnostics error.')
        self._set_diagnostics_banner(f'Dry-test error: {message}', 'error')
        self._set_diagnostics_attention(True)
        self._set_diagnostics_alert('LLM diagnostics hit an error. See Diagnostics tab.', severity='error')

    def _apply_diagnostics_results(self, results: list[dict[str, Any]], fingerprint: Optional[str], ok: bool) -> None:
        ts = time.time()
        auth_map = self._normalize_auth_entries(results, ts)
        self._diagnostics_auth_map = auth_map
        self._llm_preflight_cache = {
            'fingerprint': fingerprint if ok else None,
            'timestamp': ts,
            'status': bool(ok),
            'results': results,
            'auth_map': auth_map,
            'scope': self._diagnostics_last_scope,
        }
        self._refresh_diagnostics_auth_tree()

    def _normalize_auth_entries(self, results: list[dict[str, Any]], timestamp: float) -> dict[str, dict[str, Any]]:
        auth_map: dict[str, dict[str, Any]] = {}
        for entry in results or []:
            provider_model = entry.get('provider_model') or ''
            if not provider_model:
                provider = entry.get('provider', '') or ''
                model = entry.get('model', '') or ''
                provider_model = f"{provider}\\{model}" if provider else model
            status, message = self._map_auth_status(entry)
            auth_map[provider_model] = {
                'provider_model': provider_model,
                'provider': entry.get('provider'),
                'model': entry.get('model'),
                'status': status,
                'message': message,
                'status_code': entry.get('status_code'),
                'error_code': entry.get('error_code'),
                'tested_at': timestamp,
            }
        configured = list(dict.fromkeys(filter(None, (self.llm_models or {}).values())))
        for provider_model in configured:
            if provider_model not in auth_map:
                provider, model = _split_provider(provider_model)
                auth_map[provider_model] = {
                    'provider_model': provider_model,
                    'provider': provider,
                    'model': model,
                    'status': 'untested',
                    'message': 'Not covered by the last diagnostics run.',
                    'status_code': None,
                    'error_code': None,
                    'tested_at': timestamp,
                }
        return auth_map

    def _map_auth_status(self, result: dict[str, Any]) -> tuple[str, str]:
        if result.get('ok'):
            return 'ok', 'OK'
        status_code = result.get('status_code')
        message = self._scrub_secret(str(result.get('message') or ''))
        error_code = str(result.get('error_code') or '').lower()
        lowered = message.lower()
        if status_code == 401 or 'invalid_api_key' in lowered or error_code == 'invalid_api_key':
            return 'invalid_key', message
        if status_code in (403, 429) or 'permission' in lowered or error_code in {'permission_denied', 'access_denied'}:
            return 'rejected', message
        if status_code in (408, 504) or 'timeout' in lowered:
            return 'timeout', message
        if 'network' in lowered or 'connection' in lowered:
            return 'network_error', message
        return 'error', message

    def _auth_status_is_ok(self) -> bool:
        fingerprint = self._llm_env_fingerprint()
        if not self._is_preflight_cache_valid(fingerprint):
            return False
        cache = self._llm_preflight_cache or {}
        auth_map = cache.get('auth_map') or {}
        if not auth_map:
            return False
        return all((entry or {}).get('status') == 'ok' for entry in auth_map.values())

    def _refresh_diagnostics_env_tree(self) -> None:
        tree = getattr(self, 'diagnostics_env_tree', None)
        if not tree:
            return
        hide_unused = bool(self._diagnostics_hide_unused_var.get())
        for item in tree.get_children():
            tree.delete(item)
        for provider, env_key, status, usage in self._diagnostics_env_rows or []:
            if hide_unused and usage == 'not used':
                continue
            tree.insert('', 'end', values=(provider, env_key, status, usage))

    def _refresh_diagnostics_auth_tree(self) -> None:
        tree = getattr(self, 'diagnostics_auth_tree', None)
        if not tree:
            return
        for item in tree.get_children():
            tree.delete(item)
        order = {'ok': 0, 'invalid_key': 1, 'rejected': 1, 'network_error': 2, 'timeout': 2, 'error': 2, 'untested': 3}
        for provider_model, entry in sorted((self._diagnostics_auth_map or {}).items(), key=lambda kv: (order.get(kv[1].get('status'), 5), kv[0])):
            tested = self._format_status_timestamp(entry.get('tested_at'))
            tree.insert(
                '',
                'end',
                values=(
                    provider_model,
                    (entry.get('status') or '').upper(),
                    entry.get('status_code') or '-',
                    entry.get('message') or '',
                    tested,
                ),
            )

    def _toggle_session_keys_section(self) -> None:
        frame = getattr(self, 'diagnostics_session_frame', None)
        button = getattr(self, 'diagnostics_edit_button', None)
        if not frame or not button:
            return
        self._diagnostics_session_visible = not self._diagnostics_session_visible
        if self._diagnostics_session_visible:
            frame.pack(fill='x', padx=4, pady=(0, 8))
            button.configure(text='Hide session keys')
            self._refresh_session_entries()
        else:
            frame.pack_forget()
            button.configure(text='Edit keys (session)')

    def _session_env_keys(self, only_used_detected: bool = False) -> list[str]:
        keys: set[str] = set()
        used = self._providers_from_models()
        for provider in used:
            for group in _required_env_for_provider(provider):
                for env_key in group:
                    if not env_key:
                        continue
                    if only_used_detected and not os.environ.get(env_key):
                        continue
                    keys.add(env_key)
        if not only_used_detected:
            for envs in KNOWN_PROVIDER_ENV_VARS.values():
                for env_key in envs:
                    if env_key:
                        keys.add(env_key)
        return sorted(keys)

    def _refresh_session_entries(self) -> None:
        combo = getattr(self, 'session_key_combo', None)
        entry = getattr(self, 'session_value_entry', None)
        info_var = getattr(self, 'session_key_info_var', None)
        keys = self._session_env_keys(only_used_detected=True)
        if combo is not None:
            combo['values'] = keys
        if not keys:
            self._session_selected_key.set('')
            if combo:
                combo.set('')
                combo.configure(state='disabled')
            if entry:
                entry.configure(state='disabled')
            self._session_entry_status.set('none')
            if info_var:
                info_var.set('No detected keys mapped to current models.')
            return

        if combo:
            combo.configure(state='readonly')
        if self._session_selected_key.get() not in keys:
            self._session_selected_key.set(keys[0])
        if combo:
            combo.set(self._session_selected_key.get())
        if entry:
            entry.configure(state='normal')
            entry.focus_set()
        if info_var:
            info_var.set('Detected & used keys (select to update).')
        self._update_session_key_status()

    def _apply_session_keys(self) -> None:
        key = (self._session_selected_key.get() or '').strip()
        value = (self._session_entry_var.get() or '').strip()
        if not key:
            self.log_queue.put('[LLM] No session key selected.')
            return
        if not value:
            self.log_queue.put('[LLM] Empty value ignored.')
            return
        os.environ[key] = value
        self._injected_env_keys.add(key)
        self._session_entry_var.set('')
        self.log_queue.put(f"[LLM] Session key updated for: {key}")
        self._update_session_key_status()
        self._refresh_env_snapshot()

    def _update_session_key_status(self) -> None:
        key = (self._session_selected_key.get() or '').strip()
        entry = getattr(self, 'session_value_entry', None)
        if not key:
            self._session_entry_status.set('none')
            if entry:
                entry.configure(state='disabled')
            return
        status = 'present' if os.environ.get(key) else 'missing'
        self._session_entry_status.set(status)
        if entry:
            entry.configure(state='normal')

    def _on_session_key_selected(self, *_args) -> None:
        self._session_entry_var.set('')
        self._update_session_key_status()

    def _on_revert_session_keys(self) -> None:
        self._restore_initial_env()
        self.log_queue.put('[LLM] Reverted injected session keys to launch-time snapshot.')
        self._refresh_env_snapshot()

    def _copy_diagnostics_report(self) -> None:
        lines: list[str] = []
        lines.append(f"LLM diagnostics @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append('Env status:')
        for provider, env_key, status, usage in self._diagnostics_env_rows or []:
            lines.append(f"  - {provider} {env_key}: {status} ({usage})")
        lines.append('Auth status:')
        for provider_model, entry in (self._diagnostics_auth_map or {}).items():
            lines.append(f"  - {provider_model}: {entry.get('status')} ({entry.get('message', '')})")
        report = '\n'.join(lines)
        try:
            self.clipboard_clear()
            self.clipboard_append(report)
            self.log_queue.put('[LLM] Diagnostics report copied to clipboard.')
        except tk.TclError:
            self.log_queue.put('[LLM] Unable to copy diagnostics report to clipboard.')

    def _format_status_timestamp(self, ts: Optional[float]) -> str:
        if not ts:
            return '-'
        try:
            return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return '-'

    def _set_diagnostics_banner(self, message: str, level: str = 'info') -> None:
        self._diagnostics_banner_var.set(message)
        self._diagnostics_banner_level = level
        label = getattr(self, 'diagnostics_banner_label', None)
        if label and label.winfo_exists():
            colors = {'ok': '#166534', 'info': '#1d4ed8', 'warn': '#92400e', 'error': '#b91c1c'}
            label.configure(foreground=colors.get(level, '#1f2933'))

    def _set_diagnostics_alert(self, message: Optional[str], severity: str = 'warn') -> None:
        frame = getattr(self, 'diagnostics_alert_frame', None)
        label = getattr(self, 'diagnostics_alert_label', None)
        if not frame or not label:
            return
        if not message:
            if frame.winfo_manager():
                try:
                    frame.pack_forget()
                except tk.TclError:
                    self.log_queue.put("Unexpected error in GeneratorTab._set_diagnostics_alert")
            self._diagnostics_alert_var.set('')
            return
        colors = {'warn': '#92400e', 'error': '#b91c1c', 'info': '#1d4ed8'}
        self._diagnostics_alert_var.set(message)
        try:
            label.configure(foreground=colors.get(severity, '#92400e'))
        except tk.TclError:
            self.log_queue.put("Unexpected error in GeneratorTab._set_diagnostics_alert")
        if not frame.winfo_ismapped():
            try:
                frame.pack(fill='x', padx=8, pady=(0, 4), before=self._main_pane)
            except tk.TclError:
                self.log_queue.put("Unexpected error in GeneratorTab._set_diagnostics_alert")

    def _update_log_visibility(self, *_args) -> None:
        pane = getattr(self, "_main_pane", None)
        frame = getattr(self, "log_frame", None)
        if not pane or not frame:
            return

        panes = set(pane.panes())
        key = str(frame)

        if self._logs_visible.get():
            if key not in panes:
                pane.add(frame, weight=1)
        else:
            if key in panes:
                pane.forget(frame)

    def _toggle_logs_shortcut(self, _event=None) -> str:
        self._logs_visible.set(not self._logs_visible.get())
        self._update_log_visibility()
        return "break"

    def _set_diagnostics_attention(self, required: bool) -> None:
        self._diagnostics_attention_flag = bool(required)
        attention = self._diagnostics_attention_flag or bool(self._diagnostics_alert_var.get().strip())
        self._update_diagnostics_tab_label(attention)

    def _focus_diagnostics_tab(self) -> None:
        notebook = getattr(self, 'generator_notebook', None)
        tab = getattr(self, 'diagnostics_tab', None)
        if notebook and tab:
            try:
                notebook.select(tab)
            except tk.TclError:
                self.log_queue.put("Unexpected error in GeneratorTab._focus_diagnostics_tab")
        try:
            root = self.winfo_toplevel()
            root.deiconify()
            root.lift()
        except Exception:
             self.log_queue.put("Unexpected error in GeneratorTab._focus_diagnostics_tab")

    def _update_diagnostics_tab_label(self, attention: bool) -> None:
        notebook = getattr(self, 'generator_notebook', None)
        tab = getattr(self, 'diagnostics_tab', None)
        if not notebook or not tab:
            return
        text = 'Diagnostics LLM' + (' *' if attention else '')
        try:
            notebook.tab(tab, text=text)
        except tk.TclError:
            self.log_queue.put("Unexpected error in GeneratorTab._update_diagnostics_tab_label")

    @staticmethod
    def _scrub_secret(text: str) -> str:
        return LLMConnect._scrub_secret(text)

    def _build_widgets(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        self._outer = outer

        status_frame = ttk.Frame(outer)
        status_frame.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        log_toggle = ttk.Checkbutton(
            status_frame,
            text="Show logs (Ctrl+L)",
            variable=self._logs_visible,
            command=self._update_log_visibility,
        )
        log_toggle.pack(side="right")
        self._disabled_widget_exclusions.add(log_toggle)

        pane = ttk.Panedwindow(outer, orient="vertical")
        pane.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self._main_pane = pane

        notebook = ttk.Notebook(pane, style="Narremgen.TNotebook")
        pane.add(notebook, weight=4)
        self.generator_notebook = notebook
        self._notebook_tabs: dict[str, str] = {}

        run_frame = ttk.Frame(notebook)
        settings_frame = ttk.Frame(notebook)
        variants_frame = ttk.Frame(notebook)
        diagnostics_frame = ttk.Frame(notebook)

        notebook.add(run_frame, text="Run")
        self._notebook_tabs[notebook.tabs()[-1]] = "Run"
        notebook.add(settings_frame, text="Settings")
        self._notebook_tabs[notebook.tabs()[-1]] = "Settings"
        notebook.add(variants_frame, text="Variants")
        self._notebook_tabs[notebook.tabs()[-1]] = "Variants"
        notebook.add(diagnostics_frame, text="Diagnostics LLM")
        self._notebook_tabs[notebook.tabs()[-1]] = "Diagnostics LLM"

        self.diagnostics_tab = diagnostics_frame

        self._build_run_tab(run_frame)
        self._build_settings_tab(settings_frame)
        self._build_variants_tab(variants_frame)
        self._build_diagnostics_tab(diagnostics_frame)

        alert_frame = ttk.Frame(outer, padding=(8, 4))
        alert_label = ttk.Label(
            alert_frame,
            textvariable=self._diagnostics_alert_var,
            foreground="#92400e",
            wraplength=900,
            justify="left",
        )
        alert_label.pack(side="left", fill="x", expand=True)
        ttk.Button(
            alert_frame,
            text="Open diagnostics",
            command=self._focus_diagnostics_tab,
        ).pack(side="right", padx=(8, 0))
        self.diagnostics_alert_frame = alert_frame
        self.diagnostics_alert_label = alert_label
        log_frame = ttk.LabelFrame(pane, text="Logs")
        self.log_frame = log_frame
        ttk.Label(log_frame, textvariable=self.active_workdir_var, anchor="w").pack(fill="x", padx=6, pady=(4, 2))

        log_container = ttk.Frame(log_frame)
        log_container.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.log_text = tk.Text(log_container, wrap="word", height=15, bg="black", fg="#0f0")
        log_scroll = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.log_text.configure(state="disabled")
        self._disabled_widget_exclusions.add(self.log_text)
        self._refresh_env_snapshot()
        self._update_log_visibility()


    def _build_run_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=8)

        ttk.Label(top, text="Topic :").pack(side="left")
        entry_topic = ttk.Entry(top, textvariable=self.topic_var, width=70)
        entry_topic.pack(side="left", padx=4)
        self._controlled_widgets.append(entry_topic)

        logging_frame = ttk.LabelFrame(parent, text="Logging (GUI)")
        logging_frame.pack(fill="x", pady=4, padx=4)
        self.gui_log_level_var = tk.StringVar(value=self.gui_log_level_name)
        row = ttk.Frame(logging_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="GUI level", width=12, anchor="w").pack(side="left")
        combo = ttk.Combobox(
            row,
            textvariable=self.gui_log_level_var,
            values=["ERROR", "WARNING", "INFO", "DEBUG"],
            width=10,
            state="readonly",
        )
        combo.pack(side="left", padx=4)
        self._controlled_widgets.append(combo)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=8)
        btn_full = ttk.Button(btn_frame, text="FULL PIPELINE", command=self.run_full_pipeline)
        btn_neutral = ttk.Button(btn_frame, text="NEUTRAL ONLY", command=self.run_neutral_only)
        btn_variants = ttk.Button(btn_frame, text="VARIANTS ONLY", command=self.run_variants_only)
        # btn_stop = ttk.Button(btn_frame, text="STOP", command=self._request_stop)
        btn_full.pack(side="left", padx=4)
        btn_neutral.pack(side="left", padx=4)
        btn_variants.pack(side="left", padx=4)
        # btn_stop.pack(side="right", padx=4)
        # self._controlled_widgets.extend([btn_full, btn_neutral, btn_variants, btn_stop])
        self._controlled_widgets.extend([btn_full, btn_neutral, btn_variants])
        # self._disabled_widget_exclusions.add(btn_stop)

        variant_frame = ttk.Frame(parent)
        variant_frame.pack(fill="x", pady=8)
        ttk.Label(variant_frame, text="Variant :").pack(side="left")
        names = list(self.variants_registry.keys()) or ["direct"]
        current = self.selected_variant.get()
        if current not in names:
            current = names[0]
            self.selected_variant.set(current)
        self.variant_menu = ttk.OptionMenu(variant_frame, self.selected_variant, current, *names)
        self.variant_menu.pack(side="left", padx=4)
        self._controlled_widgets.append(self.variant_menu)

        out_frame = ttk.Frame(parent)
        out_frame.pack(fill="x", pady=8)
        ttk.Label(out_frame, text="Output dir :").pack(side="left")
        self.output_dir_var = tk.StringVar(value=str(self.output_dir))
        lbl_out = ttk.Entry(out_frame, textvariable=self.output_dir_var, width=70, state="readonly")
        lbl_out.pack(side="left", padx=4)
        btn_open = ttk.Button(out_frame, text="Open folder", command=self._open_output_dir)
        btn_open.pack(side="left", padx=4)
        self._controlled_widgets.append(btn_open)
    def _build_settings_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=8)

        paths_frame = ttk.Frame(top)
        paths_frame.pack(side="left", fill="both", expand=True, padx=4)

        self.assets_override_var = tk.StringVar(value=str(self.assets_override or ""))
        self.assets_effective_var = tk.StringVar(value=str(self.assets_dir))
        self.output_override_var = tk.StringVar(value=str(self.output_override or ""))
        self.output_effective_var = tk.StringVar(value=str(self.output_dir))

        assets_frame = ttk.LabelFrame(paths_frame, text="Assets directory")
        assets_frame.pack(fill="x", pady=(0, 8))

        assets_row = ttk.Frame(assets_frame)
        assets_row.pack(fill="x", pady=2)
        ttk.Label(assets_row, text="Assets override").pack(side="left")
        entry_assets = ttk.Entry(assets_row, textvariable=self.assets_override_var, width=45)
        entry_assets.pack(side="left", padx=4)
        btn_assets_browse = ttk.Button(assets_row, text="...", width=3, command=self._browse_assets_override)
        btn_assets_browse.pack(side="left")
        btn_assets_reset = ttk.Button(assets_row, text="Reset", command=self._reset_assets_override)
        btn_assets_reset.pack(side="left", padx=(4, 0))
        self._controlled_widgets.extend([entry_assets, btn_assets_browse, btn_assets_reset])

        assets_eff_row = ttk.Frame(assets_frame)
        assets_eff_row.pack(fill="x", pady=2)
        ttk.Label(assets_eff_row, text="Effective").pack(side="left")
        ttk.Entry(
            assets_eff_row,
            textvariable=self.assets_effective_var,
            width=45,
            state="readonly",
        ).pack(side="left", padx=4, fill="x", expand=True)

        inspect_btn = ttk.Button(assets_frame, text="Inspect assets…", command=self._inspect_assets_popup)
        inspect_btn.pack(side="left", padx=4, pady=(4, 0))
        self._controlled_widgets.append(inspect_btn)

        output_frame = ttk.LabelFrame(paths_frame, text="Output directory")
        output_frame.pack(fill="x")

        output_row = ttk.Frame(output_frame)
        output_row.pack(fill="x", pady=2)
        ttk.Label(output_row, text="Output override").pack(side="left")
        entry_output = ttk.Entry(output_row, textvariable=self.output_override_var, width=45)
        entry_output.pack(side="left", padx=4)
        btn_output_browse = ttk.Button(output_row, text="...", width=3, command=self._browse_output_override)
        btn_output_browse.pack(side="left")
        btn_output_reset = ttk.Button(output_row, text="Reset", command=self._reset_output_override)
        btn_output_reset.pack(side="left", padx=(4, 0))
        self._controlled_widgets.extend([entry_output, btn_output_browse, btn_output_reset])

        output_eff_row = ttk.Frame(output_frame)
        output_eff_row.pack(fill="x", pady=2)
        ttk.Label(output_eff_row, text="Effective").pack(side="left")
        ttk.Entry(
            output_eff_row,
            textvariable=self.output_effective_var,
            width=45,
            state="readonly",
        ).pack(side="left", padx=4, fill="x", expand=True)

        open_btn = ttk.Button(output_frame, text="Open output folder", command=self._open_output_dir)
        open_btn.pack(side="left", padx=4, pady=(4, 0))
        self._controlled_widgets.append(open_btn)

        pipe_frame = ttk.LabelFrame(top, text="Pipeline (generation)")
        pipe_frame.pack(side="left", fill="both", expand=True, padx=4)

        self.neutral_n_batches_var = tk.StringVar(value=str(self.neutral_n_batches))
        self.neutral_n_per_batch_var = tk.StringVar(value=str(self.neutral_n_per_batch))
        self.variant_batch_size_var = tk.StringVar(value=str(self.variant_batch_size))
        self.simple_max_tokens_var = tk.StringVar(value=str(self.simple_max_tokens))

        def add_int_row(frame, label, var):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")
            entry = ttk.Entry(row, textvariable=var, width=8)
            entry.pack(side="left")
            self._controlled_widgets.append(entry)

        add_int_row(pipe_frame, "Neutral batches", self.neutral_n_batches_var)
        add_int_row(pipe_frame, "Texts per batch", self.neutral_n_per_batch_var)
        add_int_row(pipe_frame, "Variant batch size", self.variant_batch_size_var)
        add_int_row(pipe_frame, "Max tokens (variants)", self.simple_max_tokens_var)

        themes_frame = ttk.LabelFrame(top, text="Themes (LLM grouping)")
        themes_frame.pack(side="left", fill="both", expand=True, padx=4)
        self.themes_min_var = tk.StringVar(value=str(self.themes_min))
        self.themes_max_var = tk.StringVar(value=str(self.themes_max))
        self.themes_batch_size_var = tk.StringVar(value=str(self.themes_batch_size))
        add_int_row(themes_frame, "Min themes", self.themes_min_var)
        add_int_row(themes_frame, "Max themes", self.themes_max_var)
        add_int_row(themes_frame, "Theme batch size", self.themes_batch_size_var)

        llm_frame = ttk.LabelFrame(parent, text="LLM models")
        llm_frame.pack(fill="x", pady=8, padx=4)
        self.llm_vars = {}
        for key in DEFAULT_LLM_MODELS.keys():
            row = ttk.Frame(llm_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{key:>18}", width=18, anchor="w").pack(side="left")
            var = tk.StringVar(value=self.llm_models.get(key, DEFAULT_LLM_MODELS[key]))
            entry = ttk.Entry(row, textvariable=var, width=40)
            entry.pack(side="left", padx=4, fill="x", expand=True)
            self.llm_vars[key] = var
            self._controlled_widgets.append(entry)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=8)
        btn_reload = ttk.Button(btn_frame, text="Reload from ini", command=self._reload_config_from_file)
        btn_save = ttk.Button(btn_frame, text="Save settings", command=self._save_settings_from_ui)
        btn_reload.pack(side="left", padx=4)
        btn_save.pack(side="right", padx=4)
        self._controlled_widgets.extend([btn_reload, btn_save])

    def _build_diagnostics_tab(self, parent):
        parent.columnconfigure(0, weight=1)

        banner = ttk.Label(
            parent,
            textvariable=self._diagnostics_banner_var,
            wraplength=960,
            justify="left",
            foreground="#92400e",
        )
        banner.pack(fill="x", padx=4, pady=(6, 8))
        self.diagnostics_banner_label = banner

        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", padx=4, pady=(0, 10))

        ttk.Button(toolbar, text="Refresh env", command=self._refresh_env_snapshot).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="Run dry-test (default)",
            command=lambda: self._start_diagnostics_test(scope="default"),
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="Test all",
            command=lambda: self._start_diagnostics_test(scope="all"),
        ).pack(side="left", padx=(0, 6))

        edit_btn = ttk.Button(toolbar, text="Edit keys (session)", command=self._toggle_session_keys_section)
        edit_btn.pack(side="left", padx=(0, 6))
        self.diagnostics_edit_button = edit_btn

        ttk.Button(toolbar, text="Copy report", command=self._copy_diagnostics_report).pack(side="right")

        env_frame = ttk.LabelFrame(parent, text="EnvStatus (presence)")
        env_frame.pack(fill="both", expand=False, padx=4, pady=(0, 10))
        env_header = ttk.Frame(env_frame)
        env_header.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(env_header, text="Known provider keys detected in the environment").pack(side="left")
        ttk.Checkbutton(
            env_header,
            text="Hide providers not mapped to LLM models",
            variable=self._diagnostics_hide_unused_var,
            command=self._refresh_diagnostics_env_tree,
        ).pack(side="right")
        env_cols = ("provider", "env_var", "status", "usage")
        env_container = ttk.Frame(env_frame)
        env_container.pack(fill="both", expand=True, padx=4, pady=4)
        env_tree = ttk.Treeview(env_container, columns=env_cols, show="headings", height=8)
        for col, width in (
            ("provider", 140),
            ("env_var", 240),
            ("status", 90),
            ("usage", 110),
        ):
            heading = "Usage" if col == "usage" else col.replace("_", " ").title()
            env_tree.heading(col, text=heading)
            env_tree.column(col, width=width, anchor="w")
        env_scroll = ttk.Scrollbar(env_container, orient="vertical", command=env_tree.yview)
        env_tree.configure(yscrollcommand=env_scroll.set)
        env_tree.pack(side="left", fill="both", expand=True)
        env_scroll.pack(side="right", fill="y")
        self.diagnostics_env_tree = env_tree

        auth_frame = ttk.LabelFrame(parent, text="AuthStatus (dry-test results)")
        auth_frame.pack(fill="both", expand=True, padx=4, pady=(0, 10))
        auth_cols = ("provider_model", "status", "code", "message", "tested_at")
        auth_container = ttk.Frame(auth_frame)
        auth_container.pack(fill="both", expand=True, padx=4, pady=4)
        auth_tree = ttk.Treeview(auth_container, columns=auth_cols, show="headings", height=8)
        for col, width in (
            ("provider_model", 260),
            ("status", 90),
            ("code", 70),
            ("message", 360),
            ("tested_at", 150),
        ):
            auth_tree.heading(col, text=col.replace("_", " ").title())
            auth_tree.column(col, width=width, anchor="w")
        auth_scroll = ttk.Scrollbar(auth_container, orient="vertical", command=auth_tree.yview)
        auth_tree.configure(yscrollcommand=auth_scroll.set)
        auth_tree.pack(side="left", fill="both", expand=True)
        auth_scroll.pack(side="right", fill="y")
        self.diagnostics_auth_tree = auth_tree

        session_frame = ttk.LabelFrame(parent, text="Session keys (process only)")
        session_container = ttk.Frame(session_frame)
        session_container.pack(fill="x", padx=4, pady=(6, 2))

        selection_row = ttk.Frame(session_container)
        selection_row.pack(fill="x", pady=(0, 6))
        ttk.Label(selection_row, text="Key").pack(side="left")
        combo = ttk.Combobox(selection_row, textvariable=self._session_selected_key, state="disabled", width=45)
        combo.pack(side="left", padx=4, fill="x", expand=True)
        combo.bind("<<ComboboxSelected>>", self._on_session_key_selected)
        self.session_key_combo = combo

        info_var = tk.StringVar(value="Detected & used keys (dropdown).")
        ttk.Label(session_container, textvariable=info_var, foreground="#475569").pack(anchor="w", padx=2, pady=(0, 4))
        self.session_key_info_var = info_var

        entry_row = ttk.Frame(session_container)
        entry_row.pack(fill="x", pady=(0, 6))
        ttk.Label(entry_row, text="New value").pack(side="left")
        entry = ttk.Entry(entry_row, textvariable=self._session_entry_var, show='•', width=48, state="disabled")
        entry.pack(side="left", padx=4, fill="x", expand=True)
        self.session_value_entry = entry
        ttk.Label(entry_row, textvariable=self._session_entry_status, foreground="#475569").pack(side="left", padx=(6, 0))

        session_frame.pack(fill="x", padx=4, pady=(0, 10))

        session_btns = ttk.Frame(session_frame)
        session_btns.pack(fill="x", padx=4, pady=(4, 8))
        ttk.Button(session_btns, text="Apply value", command=self._apply_session_keys).pack(side="left")
        ttk.Button(session_btns, text="Revert injected keys", command=self._on_revert_session_keys).pack(side="left", padx=(8, 0))

        self.diagnostics_session_frame = session_frame        
        self._set_diagnostics_banner(self._diagnostics_banner_var.get(), self._diagnostics_banner_level)

    def _browse_assets_override(self):
        initial = self.assets_override or self.assets_dir
        try:
            d = filedialog.askdirectory(initialdir=str(initial))
        except Exception:
            d = filedialog.askdirectory()
        if d:
            try:
                resolved = Path(d).expanduser().resolve()
            except Exception as exc:
                self.log_queue.put(f"[CONFIG] Invalid assets directory '{d}': {exc}")
                return
            self.assets_override = resolved
            self.assets_override_var.set(str(resolved))
            self._refresh_effective_paths()
            self._save_config()
            self.log_queue.put(f"[CONFIG] Assets override set to {resolved}")

    def _browse_output_override(self):
        initial = self.output_override or self.output_dir
        try:
            d = filedialog.askdirectory(initialdir=str(initial))
        except Exception:
            d = filedialog.askdirectory()
        if d:
            try:
                resolved = Path(d).expanduser().resolve()
            except Exception as exc:
                self.log_queue.put(f"[CONFIG] Invalid output directory '{d}': {exc}")
                return
            self.output_override = resolved
            self.output_override_var.set(str(resolved))
            self._refresh_effective_paths()
            self._save_config()
            self.log_queue.put(f"[CONFIG] Output override set to {resolved}")

    def _reset_assets_override(self):
        self.assets_override = None
        if hasattr(self, "assets_override_var"):
            self.assets_override_var.set("")
        self._refresh_effective_paths()
        self._save_config()
        self.log_queue.put("[CONFIG] Assets override cleared.")

    def _reset_output_override(self):
        self.output_override = None
        if hasattr(self, "output_override_var"):
            self.output_override_var.set("")
        self._refresh_effective_paths()
        self._save_config()
        self.log_queue.put("[CONFIG] Output override cleared.")

    def _inspect_assets_popup(self):
        assets_dir = self.assets_dir
        lines = [f"Effective assets dir: {assets_dir}", ""]
        missing = []
        for name in REQUIRED_ASSET_FILES:
            path = assets_dir / name
            status = "OK" if path.exists() else "MISSING"
            lines.append(f"- {name}: {status}")
            if status == "MISSING":
                missing.append(name)
        lines.append("")
        try:
            listing = sorted(p.name for p in assets_dir.iterdir())
            lines.append("Directory listing:")
            for item in listing[:40]:
                lines.append(f"  • {item}")
            if len(listing) > 40:
                lines.append(f"  … ({len(listing) - 40} more)")
        except Exception as exc:
            lines.append(f"(Unable to list directory: {exc})")

        status = "OK" if not missing else f"Missing files: {', '.join(missing)}"
        lines.insert(1, f"Required files check: {status}")
        messagebox.showinfo("Assets inspection", "\n".join(lines))

    @staticmethod
    def _hash_file_sha256(path: Path) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _manifest_entry(self, label: str, path: Path) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"label": label, "path": str(path), "exists": path.exists()}
        if entry["exists"] and path.is_file():
            try:
                entry["size"] = path.stat().st_size
            except OSError:
                self.log_queue.put("Unexpected error in GeneratorTab._manifest_entry")
            digest = self._hash_file_sha256(path)
            if digest:
                entry["sha256"] = digest
        return entry

    def _build_variants_tab(self, parent):
        main = ttk.Frame(parent)
        main.pack(fill="both", expand=True, padx=4, pady=8)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        columns = ("title", "active", "prompt")
        self.variants_tree = ttk.Treeview(left, columns=columns, show="headings", height=12)
        self.variants_tree.heading("title", text="Title")
        self.variants_tree.heading("active", text="Active")
        self.variants_tree.heading("prompt", text="Prompt")
        self.variants_tree.column("title", width=140, anchor="w")
        self.variants_tree.column("active", width=60, anchor="center")
        self.variants_tree.column("prompt", width=220, anchor="w")
        self.variants_tree.pack(fill="both", expand=True)
        self.variants_tree.bind("<<TreeviewSelect>>", self._on_variant_selected)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=4)
        btn_new = ttk.Button(btn_row, text="New", command=self._new_variant)
        btn_delete = ttk.Button(btn_row, text="Delete", command=self._delete_variant)
        btn_new.pack(side="left", padx=2)
        btn_delete.pack(side="left", padx=2)
        self._controlled_widgets.extend([btn_new, btn_delete])

        right = ttk.LabelFrame(main, text="Variant editor")
        right.pack(side="left", fill="both", expand=True, padx=8)

        self.edit_variant_name_var = tk.StringVar()
        self.edit_variant_title_var = tk.StringVar()
        self.edit_variant_prompt_var = tk.StringVar()
        self.edit_variant_use_sn_var = tk.BooleanVar(value=False)
        self.edit_variant_use_de_var = tk.BooleanVar(value=False)
        self.edit_variant_use_k_var = tk.BooleanVar(value=False)
        self.edit_variant_use_ops_var = tk.BooleanVar(value=False)
        self.edit_variant_active_var = tk.BooleanVar(value=True)
        self.edit_variant_kind_var = tk.StringVar(value="llm")

        def add_row(frame, label, var, width=24):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=12, anchor="w").pack(side="left")
            entry = ttk.Entry(row, textvariable=var, width=width)
            entry.pack(side="left", padx=4)
            self._controlled_widgets.append(entry)

        add_row(right, "Name (id)", self.edit_variant_name_var)
        add_row(right, "Title", self.edit_variant_title_var)

        prompt_row = ttk.Frame(right)
        prompt_row.pack(fill="x", pady=2)
        ttk.Label(prompt_row, text="Prompt", width=12, anchor="w").pack(side="left")
        entry_prompt = ttk.Entry(prompt_row, textvariable=self.edit_variant_prompt_var, width=28)
        entry_prompt.pack(side="left", padx=4)
        self._controlled_widgets.append(entry_prompt)

        def browse_prompt():
            base = self.assets_dir
            d = filedialog.askopenfilename(
                initialdir=str(base),
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if d:
                try:
                    rel = Path(d).resolve().relative_to(self.assets_dir.resolve())
                    self.edit_variant_prompt_var.set(str(rel))
                except Exception:
                    self.edit_variant_prompt_var.set(d)

        btn_browse_prompt = ttk.Button(prompt_row, text="...", width=3, command=browse_prompt)
        btn_browse_prompt.pack(side="left")
        self._controlled_widgets.append(btn_browse_prompt)

        bool_row1 = ttk.Frame(right)
        bool_row1.pack(fill="x", pady=2)
        chk_sn = ttk.Checkbutton(bool_row1, text="Use SN", variable=self.edit_variant_use_sn_var)
        chk_de = ttk.Checkbutton(bool_row1, text="Use DE", variable=self.edit_variant_use_de_var)
        chk_k = ttk.Checkbutton(bool_row1, text="Use K", variable=self.edit_variant_use_k_var)
        chk_sn.pack(side="left", padx=2)
        chk_de.pack(side="left", padx=2)
        chk_k.pack(side="left", padx=2)
        bool_row2 = ttk.Frame(right)
        bool_row2.pack(fill="x", pady=2)
        chk_ops = ttk.Checkbutton(bool_row2, text="Use ops", variable=self.edit_variant_use_ops_var)
        chk_active = ttk.Checkbutton(bool_row2, text="Active", variable=self.edit_variant_active_var)
        chk_ops.pack(side="left", padx=2)
        chk_active.pack(side="left", padx=2)
        self._controlled_widgets.extend([chk_sn, chk_de, chk_k, chk_ops, chk_active])

        kind_row = ttk.Frame(right)
        kind_row.pack(fill="x", pady=2)
        ttk.Label(kind_row, text="Kind", width=12, anchor="w").pack(side="left")
        kind_combo = ttk.Combobox(
            kind_row,
            textvariable=self.edit_variant_kind_var,
            values=["llm", "alg"],
            width=10,
            state="readonly",
        )
        kind_combo.pack(side="left", padx=4)
        self._controlled_widgets.append(kind_combo)

        btn_edit_row = ttk.Frame(right)
        btn_edit_row.pack(fill="x", pady=8)
        btn_save = ttk.Button(btn_edit_row, text="Save variant", command=self._save_variant_from_editor)
        btn_reset = ttk.Button(btn_edit_row, text="Reset editor", command=self._reset_variant_editor)
        btn_save.pack(side="left", padx=4)
        btn_reset.pack(side="left", padx=4)
        self._controlled_widgets.extend([btn_save, btn_reset])

        self._refresh_variants_tree()
    
    def _reload_config_from_file(self):
        self._load_config()
        if hasattr(self, "assets_override_var"):
            self.assets_override_var.set(str(self.assets_override or ""))
        if hasattr(self, "assets_effective_var"):
            self.assets_effective_var.set(str(self.assets_dir))
        if hasattr(self, "output_override_var"):
            self.output_override_var.set(str(self.output_override or ""))
        if hasattr(self, "output_effective_var"):
            self.output_effective_var.set(str(self.output_dir))
        if hasattr(self, "output_dir_var"):
            self.output_dir_var.set(str(self.output_dir))

        self.neutral_n_batches_var.set(str(self.neutral_n_batches))
        self.neutral_n_per_batch_var.set(str(self.neutral_n_per_batch))
        self.variant_batch_size_var.set(str(self.variant_batch_size))
        self.simple_max_tokens_var.set(str(self.simple_max_tokens))
        self.themes_min_var.set(str(self.themes_min))
        self.themes_max_var.set(str(self.themes_max))
        self.themes_batch_size_var.set(str(self.themes_batch_size))

        for key in DEFAULT_LLM_MODELS.keys():
            self.llm_vars[key].set(self.llm_models.get(key, DEFAULT_LLM_MODELS[key]))

        if hasattr(self, "gui_log_level_var"):
            self.gui_log_level_var.set(self.gui_log_level_name)
        self.log_queue.put("[CONFIG] Reloaded settings from ini.")

    def _save_settings_from_ui(self):
        def _parse_override(value: str) -> str:
            return (value or "").strip()

        assets_input = _parse_override(self.assets_override_var.get() if hasattr(self, "assets_override_var") else "")
        if assets_input:
            candidate = Path(assets_input).expanduser()
            if candidate.exists():
                self.assets_override = candidate
            else:
                self.log_queue.put(f"[CONFIG] Assets override '{candidate}' missing; reverting to packaged assets.")
                self.assets_override = None
                if hasattr(self, "assets_override_var"):
                    self.assets_override_var.set("")
        else:
            self.assets_override = None

        output_input = _parse_override(self.output_override_var.get() if hasattr(self, "output_override_var") else "")
        self.output_override = Path(output_input).expanduser() if output_input else None
        self._refresh_effective_paths()

        def parse_int(var, current, name):
            txt = var.get().strip()
            try:
                return int(txt)
            except ValueError:
                self.log_queue.put(f"[CONFIG] Invalid int for {name}='{txt}', keeping {current}")
                return current

        self.neutral_n_batches = parse_int(self.neutral_n_batches_var, self.neutral_n_batches, "neutral_n_batches")
        self.neutral_n_per_batch = parse_int(self.neutral_n_per_batch_var, self.neutral_n_per_batch, "neutral_n_per_batch")
        self.variant_batch_size = parse_int(self.variant_batch_size_var, self.variant_batch_size, "variant_batch_size")
        self.simple_max_tokens = parse_int(self.simple_max_tokens_var, self.simple_max_tokens, "simple_max_tokens")
        self.themes_min = parse_int(self.themes_min_var, self.themes_min, "themes_n_themes_min")
        self.themes_max = parse_int(self.themes_max_var, self.themes_max, "themes_n_themes_max")
        self.themes_batch_size = parse_int(self.themes_batch_size_var, self.themes_batch_size, "themes_batch_size")

        for key in DEFAULT_LLM_MODELS.keys():
            self.llm_models[key] = self.llm_vars[key].get().strip() or DEFAULT_LLM_MODELS[key]

        if hasattr(self, "gui_log_level_var"):
            self.gui_log_level_name = self.gui_log_level_var.get().upper()

        self._save_config()
        self.log_queue.put("[CONFIG] Settings saved from UI.")

    def _refresh_variants_tree(self):
        self.variants_tree.delete(*self.variants_tree.get_children())
        for name, data in sorted(self.variants_registry.items()):
            title = data.get("title", name)
            active = "yes" if data.get("active", True) else "no"
            prompt = data.get("prompt_path", "")
            self.variants_tree.insert("", "end", iid=name, values=(title, active, prompt))

    def _on_variant_selected(self, _=None):
        sel = self.variants_tree.selection()
        if not sel:
            return
        name = sel[0]
        data = self.variants_registry.get(name)
        if data:
            self._load_variant_into_editor(name, data)

    def _load_variant_into_editor(self, name, data):
        self.edit_variant_name_var.set(name)
        self.edit_variant_title_var.set(data.get("title", name))
        self.edit_variant_prompt_var.set(data.get("prompt_path", ""))
        self.edit_variant_use_sn_var.set(bool(data.get("use_sn", False)))
        self.edit_variant_use_de_var.set(bool(data.get("use_de", False)))
        self.edit_variant_use_k_var.set(bool(data.get("use_k", False)))
        self.edit_variant_use_ops_var.set(bool(data.get("use_ops", False)))
        self.edit_variant_active_var.set(bool(data.get("active", True)))
        self.edit_variant_kind_var.set(data.get("kind", "llm"))

    def _reset_variant_editor(self):
        self.edit_variant_name_var.set("")
        self.edit_variant_title_var.set("")
        self.edit_variant_prompt_var.set("")
        self.edit_variant_use_sn_var.set(False)
        self.edit_variant_use_de_var.set(False)
        self.edit_variant_use_k_var.set(False)
        self.edit_variant_use_ops_var.set(False)
        self.edit_variant_active_var.set(True)
        self.edit_variant_kind_var.set("llm")

    def _new_variant(self):
        self._reset_variant_editor()
        self.edit_variant_active_var.set(True)
        self.edit_variant_kind_var.set("llm")

    def _delete_variant(self):
        sel = self.variants_tree.selection()
        if not sel:
            return
        name = sel[0]
        if name in self.variants_registry:
            del self.variants_registry[name]
            if self.selected_variant.get() == name:
                if self.variants_registry:
                    self.selected_variant.set(next(iter(self.variants_registry)))
                else:
                    self.selected_variant.set("")
            self._save_variants_config()
            self._refresh_variants_tree()
            self._refresh_variant_menu()
            self._reset_variant_editor()
            self.log_queue.put(f"[VARIANT] Deleted variant '{name}'.")

    def _save_variant_from_editor(self):
        name = self.edit_variant_name_var.get().strip()
        if not name:
            self.log_queue.put("[VARIANT] Cannot save variant without name.")
            return
        data = {
            "name": name,
            "title": self.edit_variant_title_var.get().strip() or name,
            "prompt_path": self.edit_variant_prompt_var.get().strip(),
            "use_sn": bool(self.edit_variant_use_sn_var.get()),
            "use_de": bool(self.edit_variant_use_de_var.get()),
            "use_k": bool(self.edit_variant_use_k_var.get()),
            "use_ops": bool(self.edit_variant_use_ops_var.get()),
            "active": bool(self.edit_variant_active_var.get()),
            "kind": self.edit_variant_kind_var.get().strip() or "llm",
        }
        self.variants_registry[name] = data
        self._save_variants_config()
        self._refresh_variants_tree()
        self._refresh_variant_menu()
        if not self.selected_variant.get():
            self.selected_variant.set(name)
        self.log_queue.put(f"[VARIANT] Saved variant '{name}'.")

    def _refresh_variant_menu(self):
        menu = self.variant_menu["menu"]
        menu.delete(0, "end")
        names = list(self.variants_registry.keys()) or ["direct"]
        current = self.selected_variant.get()
        if current not in names:
            current = names[0]
            self.selected_variant.set(current)
        for name in names:
            menu.add_command(label=name, command=lambda n=name: self.selected_variant.set(n))
    
    def _set_running_ui_state(self, running: bool) -> None:
        self.is_running = running
        self.status_var.set("Running..." if running else "Idle")
        self._set_tab_enabled(enabled=not running)
        self._set_tabs_state(running)

    def _iter_tab_widgets(self):
        stack = list(self.winfo_children())
        while stack:
            widget = stack.pop()
            yield widget
            try:
                children = widget.winfo_children()
            except tk.TclError:
                children = []
            if children:
                stack.extend(children)

    def _set_tab_enabled(self, enabled: bool) -> None:
        if enabled:
            for widget, previous_state in list(self._widget_state_cache.items()):
                if not isinstance(widget, tk.Widget) or not widget.winfo_exists():
                    self._widget_state_cache.pop(widget, None)
                    continue
                try:
                    widget.configure(state=previous_state)
                except tk.TclError:
                    self.log_queue.put("Unexpected error in GeneratorTab._set_tab_enabled")
                finally:
                    self._widget_state_cache.pop(widget, None)
            return

        exclusions = getattr(self, "_disabled_widget_exclusions", set())
        for widget in self._iter_tab_widgets():
            if widget in exclusions:
                continue
            try:
                current_state = widget.cget("state")
            except tk.TclError:
                continue
            if widget not in self._widget_state_cache:
                self._widget_state_cache[widget] = current_state
            try:
                widget.configure(state="disabled")
            except tk.TclError:
                continue

    def _set_tabs_state(self, running: bool) -> None:
        notebook = getattr(self, 'generator_notebook', None)
        tabs = getattr(self, '_notebook_tabs', {})
        if not notebook or not tabs:
            return
        for tab_id, label in tabs.items():
            desired = 'normal'
            if running and label not in {'Run', 'Diagnostics LLM'}:
                desired = 'disabled'
            try:
                notebook.tab(tab_id, state=desired)
            except tk.TclError:
                continue

    def _run_async(self, work_fn: Callable[[RunContext, threading.Event], None], ctx: RunContext, success_message: str) -> None:
        if self.is_running:
            self.log_queue.put("[WARN] A pipeline run is already in progress.")
            return

        self._set_running_ui_state(True)
        cancel_event = threading.Event()
        self._current_cancel_event = cancel_event
        self._stop_requested = False
        LLMConnect.set_cancel_event(cancel_event)
        if ctx and ctx.workdir:
            self._attach_pipeline_file_handler(ctx.workdir)

        def worker():
            error: Optional[BaseException] = None
            try:
                work_fn(ctx, cancel_event)
            except Exception as exc:  # noqa: BLE001
                error = exc
            finally:
                self._completion_queue.put((ctx, error, success_message))

        thread = self._run_in_thread(worker)
        self._current_worker_thread = thread

    def _on_pipeline_finished(self, ctx: Optional[RunContext], error: Optional[BaseException], success_message: str) -> None:
        self._set_running_ui_state(False)
        self._current_cancel_event = None
        self._current_worker_thread = None
        self._stop_requested = False
        LLMConnect.set_cancel_event(None)
        self._detach_pipeline_file_handler()
        cancelled = isinstance(error, RunCancelled)
        if cancelled:
            title = "Run cancelled"
            message = str(error) or "Run cancelled by user."
        else:
            title = "Run failed" if error else "Run finished"
            message = str(error) if error else success_message
        workdir = ctx.workdir if ctx else None
        if error and not cancelled:
            self._set_diagnostics_attention(True)
            self._set_diagnostics_alert("Last run failed. Open Diagnostics LLM for details.", severity="error")
        elif self._auth_status_is_ok():
            self._set_diagnostics_alert(None)
            self._set_diagnostics_attention(False)
        if workdir is None:
            if cancelled:
                messagebox.showinfo(title, message)
            else:
                (messagebox.showerror if error else messagebox.showinfo)(title, message)
            return
        if cancelled:
            self._show_run_finished_popup(workdir=workdir, title=title, message=message)
        else:
            self._show_run_finished_popup(workdir=workdir, title=title, message=message)

    def _build_workdir_tree_text(self, workdir: Path, max_lines: int = 2000) -> str:
        workdir = Path(workdir)
        if not workdir.exists():
            return "[workdir missing]"

        lines: List[str] = []
        limit_hit = False
        lines.append("[D] .")

        def walk(current: Path) -> bool:
            nonlocal limit_hit
            try:
                entries = sorted(
                    current.iterdir(),
                    key=lambda p: (p.is_file(), p.name.lower()),
                )
            except OSError as exc:
                rel = "." if current == workdir else current.relative_to(workdir).as_posix()
                lines.append(f"[!] {rel} <error: {exc}>")
                limit_hit = True
                return False

            for entry in entries:
                rel = entry.relative_to(workdir).as_posix().replace("\\", "/")
                if entry.is_dir():
                    lines.append(f"[D] {rel}/")
                    if len(lines) >= max_lines:
                        limit_hit = True
                        return False
                    if not walk(entry):
                        return False
                else:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = "?"
                    lines.append(f"[F] {rel} ({size} bytes)")
                    if len(lines) >= max_lines:
                        limit_hit = True
                        return False
            return True

        walk(workdir)
        if not lines:
            return "[empty directory]"
        if limit_hit or len(lines) >= max_lines:
            lines.append("...truncated...")
        return "\n".join(lines)

    def _show_run_finished_popup(self, workdir: Path, title: str, message: str) -> None:
        workdir = Path(workdir)
        root = self.winfo_toplevel()
        try:
            root.deiconify()
            root.lift()
        except tk.TclError:
            self.log_queue.put("Unexpected error in GeneratorTab._show_run_finished_popup")

        popup = tk.Toplevel(self)
        popup.title(title)
        try:
            popup.transient(root)
        except tk.TclError:
            self.log_queue.put("Unexpected error in GeneratorTab._show_run_finished_popup")
        popup.grab_set()
        popup.resizable(True, True)
        popup.minsize(500, 400)

        container = ttk.Frame(popup, padding=12)
        container.pack(fill="both", expand=True)

        is_error = title.lower().startswith("run failed")
        ttk.Label(
            container,
            text=message or "",
            foreground="#b91c1c" if is_error else "#166534",
            wraplength=600,
            justify="left",
        ).pack(fill="x", pady=(0, 8))

        ttk.Label(
            container,
            text=f"Workdir: {workdir}",
            font=("Consolas", 9),
        ).pack(fill="x", pady=(0, 8))

        text_frame = ttk.Frame(container)
        text_frame.pack(fill="both", expand=True)
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        tree_text = tk.Text(text_frame, wrap="none", height=20, font=("Consolas", 10))
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=tree_text.yview)
        xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=tree_text.xview)
        tree_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree_listing = self._build_workdir_tree_text(workdir)
        tree_text.insert("1.0", tree_listing)
        tree_text.configure(state="disabled")
        tree_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, columnspan=2, sticky="ew")

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill="x", pady=(12, 0))

        def close_popup():
            try:
                popup.grab_release()
            except tk.TclError:
                self.log_queue.put("Unexpected error in GeneratorTab._show_run_finished_popup")
            popup.destroy()

        if workdir.exists():
            ttk.Button(
                btn_frame,
                text="Open folder",
                command=lambda: self._open_path_in_os(workdir),
            ).pack(side="left")

        ttk.Button(btn_frame, text="OK", command=close_popup).pack(side="right")
        popup.protocol("WM_DELETE_WINDOW", close_popup)
        popup.lift()
        popup.focus_set()
        popup.bind("<Return>", lambda _event: close_popup())

    def _poll_log_queue(self):
        if not getattr(self, "_polling_active", True):
            return

        log_widget = getattr(self, "log_text", None)
        messages: list[str] = []
        for _ in range(MAX_LOG_MSG_PER_TICK):
            try:
                raw = self.log_queue.get_nowait()
            except queue.Empty:
                break
            messages.append(str(raw).rstrip("\r\n"))

        if log_widget is not None and messages:
            try:
                _, y_last = log_widget.yview()
                auto_scroll = y_last >= 0.99
            except (tk.TclError, ValueError):
                auto_scroll = True
            block = "\n".join(messages).rstrip("\n") + "\n"
            try:
                log_widget.configure(state="normal")
                log_widget.insert("end", block)
                self._trim_log_lines()
                if auto_scroll:
                    log_widget.see("end")
            except tk.TclError:
                self.log_queue.put("Unexpected error in GeneratorTab._poll_log_queue")
            finally:
                try:
                    log_widget.configure(state="disabled")
                except tk.TclError:
                    self.log_queue.put("Unexpected error in GeneratorTab._poll_log_queue")

        try:
            while True:
                ctx, error, success_message = self._completion_queue.get_nowait()
                self._on_pipeline_finished(ctx, error, success_message)
        except queue.Empty:
            pass

        try:
            while True:
                payload = self._diagnostics_queue.get_nowait()
                self._handle_diagnostics_payload(payload)
        except queue.Empty:
            pass

        self._threads = [t for t in self._threads if t.is_alive()]
        if getattr(self, "_polling_active", True):
            try:
                self._poll_after_id = self.after(100, self._poll_log_queue)
            except tk.TclError:
                self._poll_after_id = None

    def _trim_log_lines(self) -> None:
        try:
            total_lines = int(str(self.log_text.index("end-1c")).split(".", 1)[0])
        except (ValueError, tk.TclError):
            return
        if total_lines <= self._log_line_limit:
            return
        delete_to = total_lines - self._log_line_limit
        try:
            self.log_text.delete("1.0", f"{delete_to + 1}.0")
        except tk.TclError:
            self.log_queue.put("Unexpected error in GeneratorTab._trim_log_lines")

    def _run_in_thread(self, target, *args, **kwargs):
        def runner():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = LogRedirector(self.log_queue)
            try:
                target(*args, **kwargs)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        self._threads.append(t)
        return t

    def _request_stop(self) -> None:
        cancel_event = getattr(self, "_current_cancel_event", None)
        if not self.is_running or cancel_event is None:
            self.log_queue.put("[INFO] No active run to stop.")
            return
        if self._stop_requested or cancel_event.is_set():
            self.log_queue.put("[INFO] Stop already requested; waiting for current tasks.")
            return
        cancel_event.set()
        self._stop_requested = True
        self.status_var.set("Cancelling...")
        self.log_queue.put("[INFO] Stop requested. Current stage will finish before shutting down.")

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise RunCancelled("Run cancelled by user.")

    def _install_gui_logging_handler(self) -> logging.Handler:
        handler = QueueLogHandler(self.log_queue)
        app_logger = logging.getLogger("narremgen")
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = False
        app_logger.addHandler(handler)
        return handler

    def _attach_pipeline_file_handler(self, workdir: Optional[Path]) -> None:
        if not workdir:
            return
        try:
            path = Path(workdir)
        except Exception:
            return
        self._detach_pipeline_file_handler()
        log_path = path / "pipeline.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        except Exception as exc:
            self.log_queue.put(f"[LOG] Unable to attach file handler: {exc}")
            return
        handler.setLevel(logging.INFO)
        handler.setFormatter(SensitiveFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logging.getLogger("narremgen").addHandler(handler)
        self._pipeline_file_handler = handler

    def _detach_pipeline_file_handler(self) -> None:
        handler = getattr(self, "_pipeline_file_handler", None)
        if not handler:
            return
        try:
            logging.getLogger("narremgen").removeHandler(handler)
        except Exception:
            self.log_queue.put("Unexpected error in GeneratorTab._detach_pipeline_file_handler")
        try:
            handler.close()
        except Exception:
            self.log_queue.put("Unexpected error in GeneratorTab._detach_pipeline_file_handler")
        self._pipeline_file_handler = None

    def destroy(self):
        self._polling_active = False
        after_id = getattr(self, "_poll_after_id", None)
        if after_id:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                self.log_queue.put("Unexpected error in GeneratorTab.destroy")
            self._poll_after_id = None
        self._detach_pipeline_file_handler()
        handler = getattr(self, "_log_handler", None)
        if handler:
            try:
                logging.getLogger("narremgen").removeHandler(handler)
            except Exception:
                self.log_queue.put("Unexpected error in GeneratorTab.destroy")
            self._log_handler = None
        super().destroy()

    
    def _effective_assets_dir(self):
        return self.assets_dir

    def _prepare_workdir(self, topic, create_new):
        topic_all = topic.strip()
        if not topic_all:
            raise ValueError("Empty topic.")
        topic_dir = slugify_topic(topic_all)
        workdir = get_workdir_for_topic(self.output_dir, topic_dir, create_new=create_new)
        return workdir

    def _make_algebra_path(self, workdir):
        assets = self._effective_assets_dir()
        return {
            "sn_pathdf": assets / "SN_extended_raw.csv",
            "de_pathdf": assets / "DE.csv",
            "ki_pathdf": workdir / build_csv_name(
                "context",
                "FilteredRenumeroted",
                workdir.name.rsplit("_", 1)[0],
            ),
            "opstc_pathdf": assets / "operators_structural_raw.csv",
            "opstl_pathdf": assets / "operators_stylistic_raw.csv",
        }

    def _dump_run_settings(self, workdir, topic, mode, list_variant_data, selected_variant):
        try:
            workdir = Path(workdir)
            meta_dir = workdir / "gui_parameters"
            prompts_dir = meta_dir / "prompts"
            meta_dir.mkdir(exist_ok=True)
            prompts_dir.mkdir(exist_ok=True)

            if CONFIG_PATH.exists():
                shutil.copy2(CONFIG_PATH, meta_dir / "config_gui_snapshot.ini")
            if VARIANTS_CONFIG_PATH.exists():
                shutil.copy2(VARIANTS_CONFIG_PATH, meta_dir / "variants_snapshot.ini")
            UI_CONFIG_PATH = CONFIG_ROOT / "narremgen_ui.ini"
            if UI_CONFIG_PATH.exists():
                shutil.copy2(UI_CONFIG_PATH, meta_dir / "ui_snapshot.ini")

            variants_used = []
            for vd in list_variant_data or []:
                prompt_path = vd.get("prompt_path")
                prompt_rel = ""
                prompt_resolved = ""
                prompt_sha256 = None

                if prompt_path:
                    p = Path(prompt_path)
                    try:
                        prompt_resolved = str(p.resolve())
                    except Exception:
                        prompt_resolved = str(p)

                    try:
                        prompt_rel = str(p.resolve().relative_to(self.assets_dir.resolve()))
                    except Exception:
                        prompt_rel = str(p)

                    try:
                        text = p.read_text(encoding="utf-8")
                        prompt_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        out_name = f"variant_{vd.get('name', 'unknown')}_prompt.txt"
                        (prompts_dir / out_name).write_text(text, encoding="utf-8")
                    except Exception:
                        self.log_queue.put("Unexpected error in GeneratorTab._dump_run_settings")

                variants_used.append(
                    {
                        "name": vd.get("name"),
                        "title": vd.get("title"),
                        "use_sn": bool(vd.get("use_sn")),
                        "use_de": bool(vd.get("use_de")),
                        "use_k": bool(vd.get("use_k")),
                        "use_ops": bool(vd.get("use_ops")),
                        "prompt_path": prompt_rel,
                        "prompt_resolved_path": prompt_resolved,
                        "prompt_sha256": prompt_sha256,
                    }
                )

            meta = {
                "run": {
                    "topic": topic,
                    "mode": mode,
                    #"started_at": datetime.utcnow().isoformat() + "Z",
                    "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "gui_log_level": getattr(self, "gui_log_level_name", "INFO"),
                },
                "paths": {
                    "assets_effective": str(self.assets_dir),
                    "assets_override": str(self.assets_override or ""),
                    "output_effective": str(self.output_dir),
                    "output_override": str(self.output_override or ""),
                    "packaged_assets_dir": str(self.packaged_assets_dir),
                },
                "pipeline": {
                    "neutral_n_batches": self.neutral_n_batches,
                    "neutral_n_per_batch": self.neutral_n_per_batch,
                    "variant_batch_size": self.variant_batch_size,
                    "simple_max_tokens": self.simple_max_tokens,
                },
                "themes": {
                    "n_themes_min": self.themes_min,
                    "n_themes_max": self.themes_max,
                    "themes_batch_size": self.themes_batch_size,
                },
                "llm": self.llm_models,
                "variants": {
                    "selected": selected_variant or "",
                    "used": variants_used,
                },
            }

            out_meta = meta_dir / "run_metadata.json"
            out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log_queue.put(f"[META] Saved GUI run parameters -> {out_meta}")

            resolved = {
                "effective": {
                    "assets_dir": str(self.assets_dir),
                    "output_dir": str(self.output_dir),
                },
                "overrides": {
                    "assets_override": str(self.assets_override or ""),
                    "output_override": str(self.output_override or ""),
                },
                "packaged_assets_dir": str(self.packaged_assets_dir),
            }
            resolved_path = meta_dir / "resolved_paths.json"
            resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log_queue.put(f"[META] Saved resolved paths -> {resolved_path}")

            manifest_entries: List[Dict[str, Any]] = []
            for name in REQUIRED_ASSET_FILES:
                manifest_entries.append(self._manifest_entry(f"asset:{name}", self.assets_dir / name))
            algebra_paths = self._make_algebra_path(workdir)
            for key, path_obj in algebra_paths.items():
                manifest_entries.append(self._manifest_entry(f"algebra:{key}", Path(path_obj)))
            manifest_path = meta_dir / "assets_manifest.json"
            manifest_path.write_text(json.dumps(manifest_entries, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log_queue.put(f"[META] Saved assets manifest -> {manifest_path}")
        except Exception as exc:
            self.log_queue.put(f"[META] Failed to dump run settings: {exc}")
            self.log_queue.put(traceback.format_exc())

    def _get_variant_list(self, selected_variant: Optional[str] = None):
        if selected_variant is not None:
            name = selected_variant
        else:
            name = self.selected_variant.get()
        data = self.variants_registry.get(name)
        if not data:
            return []
        prompt_path = data.get("prompt_path", "")
        prompt_full_path = None
        if prompt_path:
            p = Path(prompt_path)
            if not p.is_absolute():
                base = self.assets_dir
                p = base / p
            prompt_full_path = p

        variant_data = {
            "name": data["name"],
            "title": data.get("title", data["name"]),
            "prompt_path": prompt_full_path,
            "use_sn": bool(data.get("use_sn", False)),
            "use_de": bool(data.get("use_de", False)),
            "use_k": bool(data.get("use_k", False)),
            "use_ops": bool(data.get("use_ops", False)),
        }
        return [variant_data]

    
    def run_neutral_only(self):
        topic = self.topic_var.get().strip()
        selected_variant_value = self.selected_variant.get()
        if not topic:
            return
        if not self.ensure_llm_ready_or_route_to_diagnostics():
            return

        workdir: Optional[Path] = None
        self._set_running_ui_state(True)
        self.update_idletasks()
        try:
            self._refresh_effective_paths()
            workdir = self._prepare_workdir(topic, create_new=True)
            if workdir:
                self.active_workdir_var.set(f"Workdir: {workdir}")
        except Exception as exc:
            self.log_queue.put(f"[ERROR] Failed to prepare workdir: {exc}")
            self.log_queue.put(traceback.format_exc())
            messagebox.showerror("Run failed", f"Could not prepare workdir:\n{exc}")
            self.active_workdir_var.set("Workdir: -")
            return
        finally:
            self._set_running_ui_state(False)

        ctx = RunContext(
            topic=topic,
            workdir=workdir,
            mode="neutral_only",
            selected_variant=selected_variant_value,
            list_variant_data=tuple(),
        )

        def job(run_ctx: RunContext, cancel_event: threading.Event):
            self.log_queue.put(f"[INFO] NEUTRAL ONLY for topic={run_ctx.topic}")
            self.log_queue.put(f"[INFO] WORKDIR = {run_ctx.workdir}")
            try:
                self._raise_if_cancelled(cancel_event)
                self._dump_run_settings(
                    workdir=run_ctx.workdir,
                    topic=run_ctx.topic,
                    mode=run_ctx.mode,
                    list_variant_data=run_ctx.list_variant_data,
                    selected_variant=run_ctx.selected_variant,
                )
                self._raise_if_cancelled(cancel_event)
                run_pipeline(
                    topic=run_ctx.topic,
                    workdir=run_ctx.workdir,
                    assets_dir=self._effective_assets_dir(),
                    n_batches=self.neutral_n_batches,
                    n_per_batch=self.neutral_n_per_batch,
                    output_format="txt",
                    dialogue_mode="single",
                    verbose=True,
                )
                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[INFO] Neutral pipeline done.")

                if run_ctx.workdir is None:
                    raise RuntimeError("workdir is None (prepare_workdir failed)")

                try:
                    load_neutral_data(run_ctx.workdir, verbose=True)
                except Exception as e:
                    self.log_queue.put(f"[NEUTRAL] Failed to materialize files for neutral in variants/")

            except RunCancelled:
                raise
            except LLMCancellation as exc:
                raise RunCancelled(str(exc)) from exc
            except Exception as exc:
                self.log_queue.put(f"[ERROR] Neutral pipeline failed: {exc}")
                self.log_queue.put(traceback.format_exc())
                raise

        self._run_async(job, ctx, success_message="Neutral pipeline completed.")

    def run_variants_only(self):
        topic = self.topic_var.get().strip()
        selected_variant_value = self.selected_variant.get()
        if not topic:
            return
        if not self.ensure_llm_ready_or_route_to_diagnostics():
            return

        workdir: Optional[Path] = None
        list_variant_data: Optional[list[Dict[str, Any]]] = None
        self._set_running_ui_state(True)
        self.update_idletasks()
        try:
            self._refresh_effective_paths()
            workdir = self._prepare_workdir(topic, create_new=False)
            if workdir:
                self.active_workdir_var.set(f"Workdir: {workdir}")
            list_variant_data = self._get_variant_list(selected_variant_value)
        except Exception as exc:
            self.log_queue.put(f"[ERROR] Failed to prepare variants run: {exc}")
            self.log_queue.put(traceback.format_exc())
            messagebox.showerror("Run failed", f"Could not prepare variants run:\n{exc}")
            self.active_workdir_var.set("Workdir: -")
            return
        finally:
            self._set_running_ui_state(False)

        variant_tuple = tuple(list_variant_data or [])

        ctx = RunContext(
            topic=topic,
            workdir=workdir,
            mode="variants_only",
            selected_variant=selected_variant_value,
            list_variant_data=variant_tuple,
        )

        def job(run_ctx: RunContext, cancel_event: threading.Event):
            self.log_queue.put(f"[INFO] VARIANTS ONLY for topic={run_ctx.topic}")
            self.log_queue.put(f"[INFO] WORKDIR = {run_ctx.workdir}")
            try:
                self._raise_if_cancelled(cancel_event)
                algebra_path = self._make_algebra_path(run_ctx.workdir)
                variant_list = run_ctx.list_variant_data or []
                if not variant_list:
                    self.log_queue.put("[WARN] No variant selected.")
                    return

                self._dump_run_settings(
                    workdir=run_ctx.workdir,
                    topic=run_ctx.topic,
                    mode=run_ctx.mode,
                    list_variant_data=variant_list,
                    selected_variant=run_ctx.selected_variant,
                )

                overwrite = bool(getattr(self, "variants_overwrite_existing", True))
                for variant_data in variant_list:
                    self._raise_if_cancelled(cancel_event)
                    name = variant_data["name"]
                    self.log_queue.put(f"[INFO] Variant {name} (overwrite={overwrite})")
                    try:
                        out_variant = run_one_variant_pipeline(
                            workdir=run_ctx.workdir,
                            topic=run_ctx.topic,
                            variant_data=variant_data,
                            algebra_path=algebra_path,
                            max_tokens=self.simple_max_tokens,
                            batch_size=self.variant_batch_size,
                            verbose=True,
                            overwrite_existing=overwrite,
                            do_stats=True,
                        )
                        self.log_queue.put(f"[OK] Variant {name} done: {out_variant}")
                    except Exception as exc:
                        self.log_queue.put(f"[ERROR] Variant {name} failed: {exc}")
                        self.log_queue.put(traceback.format_exc())

                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[INFO] Variants phase done.")
            except RunCancelled:
                raise
            except LLMCancellation as exc:
                raise RunCancelled(str(exc)) from exc
            except Exception as exc:
                self.log_queue.put(f"[ERROR] Variants-only pipeline failed: {exc}")
                self.log_queue.put(traceback.format_exc())
                raise

        self._run_async(job, ctx, success_message="Variants pipeline completed.")

    def run_full_pipeline(self):
        topic = self.topic_var.get().strip()
        selected_variant_value = self.selected_variant.get()
        if not topic:
            return
        if not self.ensure_llm_ready_or_route_to_diagnostics():
            return

        workdir: Optional[Path] = None
        list_variant_data: Optional[list[Dict[str, Any]]] = None
        self._set_running_ui_state(True)
        self.update_idletasks()
        try:
            self._refresh_effective_paths()
            workdir = self._prepare_workdir(topic, create_new=True)
            if workdir:
                self.active_workdir_var.set(f"Workdir: {workdir}")
            list_variant_data = self._get_variant_list(selected_variant_value)
        except Exception as exc:
            self.log_queue.put(f"[ERROR] Failed to prepare full pipeline: {exc}")
            self.log_queue.put(traceback.format_exc())
            messagebox.showerror("Run failed", f"Could not prepare full pipeline:\n{exc}")
            self.active_workdir_var.set("Workdir: -")
            return
        finally:
            self._set_running_ui_state(False)

        variant_tuple = tuple(list_variant_data or [])

        ctx = RunContext(
            topic=topic,
            workdir=workdir,
            mode="full",
            selected_variant=selected_variant_value,
            list_variant_data=variant_tuple,
        )

        def job(run_ctx: RunContext, cancel_event: threading.Event):
            self.log_queue.put(f"[INFO] FULL PIPELINE for topic={run_ctx.topic}")
            self.log_queue.put(f"[INFO] WORKDIR = {run_ctx.workdir}")
            try:
                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[STEP] Neutral generation…")
                run_pipeline(
                    topic=run_ctx.topic,
                    workdir=run_ctx.workdir,
                    assets_dir=self._effective_assets_dir(),
                    n_batches=self.neutral_n_batches,
                    n_per_batch=self.neutral_n_per_batch,
                    output_format="txt",
                    dialogue_mode="single",
                    verbose=True,
                )

                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[STEP] Variants generation…")
                algebra_path = self._make_algebra_path(run_ctx.workdir)
                variant_list = run_ctx.list_variant_data or []
                self._dump_run_settings(
                    workdir=run_ctx.workdir,
                    topic=run_ctx.topic,
                    mode=run_ctx.mode,
                    list_variant_data=variant_list,
                    selected_variant=run_ctx.selected_variant,
                )

                if not variant_list:
                    self.log_queue.put("[WARN] No variant selected.")
                else:
                    overwrite = bool(getattr(self, "variants_overwrite_existing", True))
                    for variant_data in variant_list:
                        self._raise_if_cancelled(cancel_event)
                        name = variant_data["name"]
                        self.log_queue.put(f"[INFO] Variant {name} (overwrite={overwrite})")
                        try:
                            out_variant = run_one_variant_pipeline(
                                workdir=run_ctx.workdir,
                                topic=run_ctx.topic,
                                variant_data=variant_data,
                                algebra_path=algebra_path,
                                max_tokens=self.simple_max_tokens,
                                batch_size=self.variant_batch_size,
                                verbose=True,
                                overwrite_existing=overwrite,
                                do_stats=True,
                            )
                            self.log_queue.put(f"[OK] Variant {name} done: {out_variant}")
                        except Exception as exc:
                            self.log_queue.put(f"[ERROR] Variant {name} failed: {exc}")
                            self.log_queue.put(traceback.format_exc())

                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[STEP] Themes classification…")
                THEMES_JSON_NAME = "advice_groups_llm.json"
                THEMES_ASSIGNMENT_JSON_NAME = "advice2groups_llm.json"
                themes_group_path = run_ctx.workdir / "themes" / THEMES_JSON_NAME
                themes_assign_path = run_ctx.workdir / "themes" / THEMES_ASSIGNMENT_JSON_NAME

                if not themes_group_path.exists() or not themes_assign_path.exists():
                    self.log_queue.put("[INFO] No themes JSON found, launching LLM theme pipeline.")

                MAX_RETRIES = 5
                attempt = 0
                while attempt < MAX_RETRIES:
                    self._raise_if_cancelled(cancel_event)
                    try:
                        self.log_queue.put(f"[THEMES] Launch attempt {attempt + 1}/{MAX_RETRIES}…")
                        run_llm_theme_pipeline(
                            workdir=run_ctx.workdir,
                            themes_json_name=THEMES_JSON_NAME,
                            assignments_json_name=THEMES_ASSIGNMENT_JSON_NAME,
                            n_themes_min=self.themes_min,
                            n_themes_max=self.themes_max,
                            batch_size=self.themes_batch_size,
                            verbose=True,
                        )
                        self.log_queue.put("[THEMES] Success.")
                        break
                    except Exception as exc:
                        attempt += 1
                        self.log_queue.put(f"[THEMES] ERROR (attempt {attempt}): {exc}")
                        self.log_queue.put(traceback.format_exc())
                        if attempt >= MAX_RETRIES:
                            self.log_queue.put("[THEMES] Max retries reached. Aborting themes step.")
                        else:
                            self.log_queue.put("[THEMES] Retrying…")

                try:
                    themes_dir = run_ctx.workdir / "themes"
                    assignments_json = themes_dir / "advice2groups_llm.json"
                    chapters_json = themes_dir / "chapters_llm.json"
                    themes_json = themes_dir / "advice_groups_llm.json"

                    if assignments_json.is_file():
                        self.log_queue.put("[STEP] Chapters (from themes)…")
                        create_chapters_from_theme_assignments(
                            assignments_json=str(assignments_json),
                            out_json=str(chapters_json),
                            merge_small_chapters=0,
                            small_chapters_title="Other",
                        )
                        self.log_queue.put("[CHAPTERS] Success.")

                        self.log_queue.put("[STEP] TeX export (chapters)…")

                        document_title = str(run_ctx.topic) if run_ctx.topic else run_ctx.workdir.name
                        build_merged_tex_from_csv(
                            workdir=str(run_ctx.workdir),
                            document_title=document_title,
                            variant_name="neutral",
                            output_tex=None,
                            chapters_json=str(chapters_json),
                            themes_json=str(themes_json) if themes_json.is_file() else None,
                            show_entry_numbers=False,
                            show_snde=True,
                            show_chapter_numbers=False,
                        )
                        vn = str(variant_data.get("name", "")).strip().lower()
                        if vn and vn != "neutral":
                            build_merged_tex_from_csv(
                                workdir=str(run_ctx.workdir),
                                document_title=document_title,
                                variant_name=vn, #variant_data["name"].lower(),
                                output_tex=None,
                                chapters_json=str(chapters_json),
                                themes_json=str(themes_json) if themes_json.is_file() else None,
                                show_entry_numbers=False,
                                show_snde=True,
                                show_chapter_numbers=False,
                            )                        
                        self.log_queue.put("[TEX] Success.")
                    else:
                        self.log_queue.put("[CHAPTERS] Skipped (no themes assignments json).")
                except Exception as e:
                    self.log_queue.put(f"[TEX] Failed")


                self.log_queue.put("[STEP] Global stats across variants…")
                stats_files = {}
                for variant_data in variant_list:
                    self._raise_if_cancelled(cancel_event)
                    vt = variant_data["name"].lower()
                    stats_path = run_ctx.workdir / "variants" / vt / f"stats_neutral_vs_{vt}.csv"
                    if stats_path.exists():
                        stats_files[vt] = stats_path
                        self.log_queue.put(f"[DEBUG] Found stats for {vt}: {stats_path}")
                    else:
                        self.log_queue.put(f"[WARN] Missing stats file for: {vt}")

                if not stats_files:
                    self.log_queue.put("[WARN] No stats files found — skipping global table.")
                else:
                    stats_dict = {}
                    for vt, path in stats_files.items():
                        df = pd.read_csv(path, sep=";")
                        self.log_queue.put(f"[DEBUG] Loaded stats {vt}, shape={df.shape}")
                        stats_dict[vt] = df

                    global_stats = build_global_stats_table(
                        stats_dict=stats_dict,
                        list_variants=list(stats_files.keys()),
                        metric_source="both",
                        verbose=True,
                    )

                    if global_stats is not None:
                        out_csv = run_ctx.workdir / "stats_all_variants_table.csv"
                        global_stats.to_csv(out_csv, sep=";", index=True)
                        self.log_queue.put(f"[STATS] Saved CSV -> {out_csv}")
                        #out_tex = run_ctx.workdir / "stats_all_variants_table.tex"
                        #tex_code = global_stats.to_latex(index=True, escape=False)
                        #out_tex.write_text(tex_code, encoding="utf-8")
                        #self.log_queue.put(f"[STATS] Saved TEX -> {out_tex}")
                    else:
                        self.log_queue.put("[WARN] Global stats not built — nothing to save.")

                self._raise_if_cancelled(cancel_event)
                self.log_queue.put("[INFO] FULL PIPELINE DONE.")
            except RunCancelled:
                raise
            except LLMCancellation as exc:
                raise RunCancelled(str(exc)) from exc
            except Exception as exc:
                self.log_queue.put(f"[ERROR] Full pipeline failed: {exc}")
                self.log_queue.put(traceback.format_exc())
                raise

        self._run_async(job, ctx, success_message="Full pipeline completed.")

    def _open_path_in_os(self, path: Optional[Path]) -> None:
        if not path:
            return
        try:
            resolved = Path(path)
        except TypeError:
            resolved = Path(str(path))
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(resolved))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(resolved)], check=False)
            else:
                subprocess.run(["xdg-open", str(resolved)], check=False)
        except Exception as exc:
            self.log_queue.put(f"[OS] Could not open folder {resolved}: {exc}")

    def _open_output_dir(self):
        self._open_path_in_os(self.output_dir)

class Dashboard(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg='#333333', padx=10, pady=5, **kwargs)
        self.lbl_main = tk.Label(self, text="PRÊT", font=('Arial', 14, 'bold'), bg='#333333', fg='white')
        self.lbl_main.pack(side='left')
        self.lbl_sub = tk.Label(self, text="-", font=('Arial', 10), bg='#333333', fg='#cccccc')
        self.lbl_sub.pack(side='right')

    def update_stats(self, stats):
        pct = 0
        if stats['total_texts'] > 0: pct = (stats['count_selected'] / stats['total_texts']) * 100
        self.lbl_main.config(text=f"SÉLECTION : {stats['count_selected']} / {stats['total_texts']} ({pct:.1f}%)")
        self.lbl_sub.config(text=f"Filtre Actif : {stats['count_filtered']} textes visibles")

class NavigationPanel(tk.Frame):
    def __init__(self, master, callbacks, **kwargs):
        super().__init__(master, padx=6, pady=6, bg='#f4f5f7', **kwargs)
        self.callbacks = callbacks
        self.locked = False
        self.show_codes = tk.BooleanVar(value=False)
        self._codes_suffix_re = re.compile(
            r"\s*\((?:\s*(?:SN|DE)\d+[A-Za-z]*\s*(?:,\s*(?:SN|DE)\d+[A-Za-z]*)*)\)\s*$",
            re.IGNORECASE,
        )
        self.font_fam = CFG.get('UI', 'font_family')
        base_font_sz = _load_ui_font_size('ui_font_size_nav', 'font_size_nav', 'font_size_list')
        self.font_sz = base_font_sz
        self._font_size_var = tk.IntVar(value=base_font_sz)
        self.row_ht = max(20, base_font_sz + 8)
        self._tree_font = tkfont.Font(family=self.font_fam, size=self.font_sz)
        self._nav_style = ttk.Style()

        f_search = tk.Frame(self, bg=self['bg'])
        f_search.pack(fill='x', pady=2)
        tk.Label(f_search, text="Search:", bg=self['bg']).pack(side='left')
        self.entry_search = tk.Entry(f_search, font=(self.font_fam, self.font_sz))
        self.entry_search.pack(side='left', fill='x', expand=True, padx=(4, 0))
        self.entry_search.bind('<KeyRelease>', lambda e: self.callbacks['search'](self.entry_search.get()))
        self.chk_codes = tk.Checkbutton(
            f_search,
            text="Codes",
            variable=self.show_codes,
            bg=self['bg'],
            command=self._refresh,
        )
        tk.Label(f_search, text="Font:", bg=self['bg']).pack(side='left', padx=(6, 0))
        size_box = ttk.Combobox(
            f_search,
            width=4,
            state="readonly",
            values=[str(val) for val in FONT_SIZE_CHOICES],
            textvariable=self._font_size_var,
        )
        size_box.pack(side='left')
        self._font_size_var.trace_add('write', lambda *_: self._apply_font_size())

        self._nav_style.configure(
            "Nav.Treeview",
            font=(self.font_fam, self.font_sz),
            rowheight=self.row_ht,
            background="#fdfdfd",
            fieldbackground="#fdfdfd",
            bordercolor="#d0d0d0",
            relief="flat",
        )
        self._nav_style.map(
            "Nav.Treeview",
            background=[('selected', '#dbeafe')],
            foreground=[('selected', '#0f172a')],
        )

        tree_frame = tk.Frame(self, bg=self['bg'])
        tree_frame.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse", style="Nav.Treeview")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        sb.pack(side='right', fill='y')
        sbx = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        sbx.pack(side='bottom', fill='x')
        self.tree.configure(yscrollcommand=sb.set, xscrollcommand=sbx.set)
        self.tree.pack(side='left', fill='both', expand=True)
        self.tree.column("#0", anchor='w', stretch=True, width=240, minwidth=180)
        self.tree.heading("#0", text="Textes filtrés")

        self.tree.tag_configure('selected', background='#dbeafe', foreground='#0f172a')
        self.tree.tag_configure('normal', background='#fdfdfd', foreground='#1f2933')
        self.tree.tag_configure('current', background='#fff7e6', foreground='#1f2933', font=(self.font_fam, self.font_sz, 'bold'))

        self.tree.bind('<<TreeviewSelect>>', self._on_select)
        self.tree.bind('<space>', self._on_space)
        self.tree.bind('<Double-1>', self._on_space)
        self.last_data = []
        self.last_idx = -1
        self._apply_font_size()

    def update_list(self, data, idx):
        self.last_data = data
        self.last_idx = idx
        self._refresh()

    def _refresh(self):
        self.locked = True
        self.tree.delete(*self.tree.get_children())
        sel_id = None
        show = self.show_codes.get()
        display_texts = []
        for i, h, is_sel, ex in self.last_data:
            pre = "[x] " if is_sel else "[ ] "
            header = self._strip_trailing_codes(h or "")
            txt = f"{pre}{header}" + (ex if show else "")
            display_texts.append(txt)
            tags = ['selected'] if is_sel else ['normal']
            if i == self.last_idx:
                tags.append('current')
            iid = self.tree.insert("", "end", iid=str(i), text=txt, tags=tags)
            if i == self.last_idx:
                sel_id = iid
        if sel_id:
            try:
                self.tree.selection_set(sel_id)
                self.tree.see(sel_id)
            except Exception:
                _logger.warning("Unexpected error in NavigationPanel._refresh", exc_info=True)
        if display_texts:
            widest = max(self._tree_font.measure(t) for t in display_texts) + 40
            target = min(900, max(200, widest))
            self.tree.column("#0", width=target)
        self.after_idle(lambda: setattr(self, 'locked', False))

    def _apply_font_size(self):
        try:
            requested = int(self._font_size_var.get())
        except (TypeError, ValueError, tk.TclError):
            return
        normalized = _clamp_font_size(requested, minimum=9, maximum=26)
        if normalized != requested:
            self._font_size_var.set(normalized)
            return
        if normalized == self.font_sz:
            return
        self.font_sz = normalized
        self.row_ht = max(20, normalized + 8)
        self._tree_font.configure(size=normalized)
        try:
            self.entry_search.configure(font=(self.font_fam, normalized))
        except tk.TclError:
            _logger.warning("Unexpected error in NavigationPanel._apply_font_size", exc_info=True)
        self.tree.tag_configure('current', font=(self.font_fam, normalized, 'bold'))
        try:
            self._nav_style.configure("Nav.Treeview", font=(self.font_fam, normalized), rowheight=self.row_ht)
        except tk.TclError:
            _logger.warning("Unexpected error in NavigationPanel._apply_font_size", exc_info=True)
        try:
            CFG.set('UI', 'ui_font_size_nav', normalized)
        except Exception:
            _logger.warning("Unexpected error in NavigationPanel._apply_font_size", exc_info=True)


    def _strip_trailing_codes(self, header: str) -> str:
        if not header:
            return header
        stripped = self._codes_suffix_re.sub("", header).rstrip()
        stripped = stripped.rstrip("-–— ").rstrip()
        return stripped or header

    def _on_select(self, _event):
        if self.locked:
            return
        try:
            self.callbacks['nav_to'](int(self.tree.selection()[0]))
            self.tree.focus_set()
        except Exception:
            _logger.warning("Unexpected error in NavigationPanel._on_select", exc_info=True)

    def _on_space(self, _event):
        self.callbacks['toggle_sel']()
        return "break"

class VariantView(tk.Frame):
    def __init__(self, master, slot, callbacks, font_size_var: Optional[tk.IntVar] = None, **kwargs):
        super().__init__(master, bd=2, relief=tk.GROOVE, **kwargs)
        self.slot = slot; self.callbacks = callbacks
        header_bg = CFG.get('COLORS','header_bg')
        self._header_base_bg = header_bg
        self.header = tk.Frame(self, bg=header_bg, padx=2)
        self.header.pack(fill='x')
        self.lbl_f = tk.Label(self.header, text="-", font=('Arial', 8, 'italic'), bg=header_bg)
        self.lbl_f.pack(side='top', anchor='w')
        self.lbl_t = tk.Label(self.header, text=f"Vue {slot+1}", font=('Arial', 10, 'bold'), bg=header_bg)
        self.lbl_t.pack(side='left', fill='x', expand=True)
        bf = tk.Frame(self.header, bg=header_bg)
        bf.pack(side='right')
        def btn(t, c):
            tk.Button(bf, text=t, command=c, width=3).pack(side='left', padx=1)
        btn("📋", lambda: callbacks['copy_one'](slot))
        btn("💾", lambda: callbacks['save_one'](slot))
        btn("📂", lambda: callbacks['load'](slot))
        btn("🗑", lambda: callbacks['clear'](slot))
        self._font_family = CFG.get('UI','font_family') or 'Consolas'
        if font_size_var is None:
            font_size_var = tk.IntVar(value=_load_ui_font_size('ui_font_size_variants', 'ui_font_size_main', 'font_size_text'))
        self._font_var = font_size_var
        initial_size = _clamp_font_size(self._font_var.get())
        if initial_size != self._font_var.get():
            self._font_var.set(initial_size)
        selector = ttk.Combobox(
            bf,
            width=4,
            state="readonly",
            values=[str(val) for val in FONT_SIZE_CHOICES],
            textvariable=self._font_var,
        )
        selector.pack(side='left', padx=(6, 0))
        self._font_trace_id = self._font_var.trace_add('write', lambda *_: self._sync_text_font())
        self._text_font = tkfont.Font(family=self._font_family, size=initial_size)
        self.txt = scrolledtext.ScrolledText(self, wrap=tk.WORD, state=tk.DISABLED, font=self._text_font)
        self.txt.pack(fill='both', expand=True, padx=5, pady=5)
        for w in [self.txt, self.lbl_t, self.header]:
            w.bind('<Button-1>', lambda e: callbacks['toggle']())

    def update_view(self, data):
        self.txt.config(state=tk.NORMAL)
        self.txt.delete('1.0', tk.END)
        if not data:
            self.lbl_f.config(text="Empty", fg='red')
            self.lbl_t.config(text="Empty/Filtered")
            self.txt.insert('1.0', "...")
            self.config(bg='#f5f5f5')
            self.header.config(bg=self._header_base_bg)
        else:
            self.lbl_f.config(text=data['filename'], fg='black')
            self.lbl_t.config(text=data['header'])
            self.txt.insert('1.0', data['text'])
            c = CFG.get('COLORS','bg_selected') if data['is_selected'] else '#f5f5f5'
            self.config(bg=c)
            self.header.config(bg='#99c2ff' if data['is_selected'] else self._header_base_bg)
        self.txt.config(state=tk.DISABLED)

    def _sync_text_font(self):
        try:
            size = int(self._font_var.get())
        except (TypeError, ValueError, tk.TclError):
            return
        normalized = _clamp_font_size(size)
        if normalized != size:
            self._font_var.set(normalized)
            return
        self._text_font.configure(size=normalized)

class StatsPanel(tk.Frame):
    def __init__(self, master, callbacks, **kwargs):
        super().__init__(master, **kwargs)
        self.callbacks = callbacks
        self.nb = ttk.Notebook(self); self.nb.pack(fill='both', expand=True)
        self.f_list = tk.Frame(self.nb); self.nb.add(self.f_list, text="Filters")
        self.f_gauge = tk.Frame(self.nb); self.nb.add(self.f_gauge, text="Plots")
        base_family = CFG.get('UI', 'font_family') or 'Consolas'
        base_size = int(CFG.get('UI', 'font_size_list'))
        self.list_font = tkfont.Font(family=base_family, size=base_size)
        self.font_size_var = tk.IntVar(value=base_size)
        self.show_percent = tk.BooleanVar(value=False)
        self._setup_list(); self._setup_gauge()
        self.font_size_var.trace_add('write', lambda *_: self._apply_font_size())
        self._apply_font_size()
        self.last_stats = None
        self._code_pattern = re.compile(r'([A-Za-z]+)(\d+)(.*)')

    def _setup_list(self):
        ctrl = tk.Frame(self.f_list)
        ctrl.pack(fill='x', pady=(4, 0))
        tk.Label(ctrl, text="Taille police:", font=('Arial', 8)).pack(side='left')
        tk.Spinbox(
            ctrl,
            from_=8,
            to=18,
            width=4,
            textvariable=self.font_size_var
        ).pack(side='left', padx=(4, 12))
        tk.Checkbutton(
            ctrl,
            text="Afficher %",
            variable=self.show_percent,
            command=lambda: self.update_stats(self.last_stats)
        ).pack(side='left')
        paned = tk.PanedWindow(self.f_list, orient=tk.VERTICAL); paned.pack(fill='both', expand=True)
        self.txt_sn = self._mk_pane(paned, "SN", 'sn'); self.txt_de = self._mk_pane(paned, "DE", 'de')

    def _mk_pane(self, parent, title, key):
        f = tk.Frame(parent); parent.add(f, minsize=100)
        h = tk.Frame(f); h.pack(fill='x')
        tk.Label(h, text=f"Filtres {title}", font=('Arial', 8, 'bold')).pack(side='left')
        tk.Button(h, text="None", font=('Arial',7), command=lambda: self.callbacks[f'none_{key}'](), bg='#ffcdd2').pack(side='right')
        tk.Button(h, text="All", font=('Arial',7), command=lambda: self.callbacks[f'all_{key}'](), bg='#c8e6c9').pack(side='right')
        t = scrolledtext.ScrolledText(f, width=30, height=10, font=self.list_font, cursor="arrow")
        t.pane_key = key
        t.pack(fill='both', expand=True); t.bind("<Button-1>", lambda e, k=title: self._click(e, k, t))
        self._tags(t); return t

    def _setup_gauge(self):
        self._gauge_sash_init = False

        paned = tk.PanedWindow(self.f_gauge, orient=tk.VERTICAL, sashwidth=8)
        paned.pack(fill='both', expand=True)
        self._gauge_paned = paned

        sn_frame = tk.Frame(paned)
        de_frame = tk.Frame(paned)

        paned.add(sn_frame, stretch='always', minsize=160)
        paned.add(de_frame, stretch='always', minsize=70)

        tk.Label(sn_frame, text="SN (selected in current list)", font=('Arial', 8, 'bold')).pack(anchor='w', padx=4, pady=(4, 0))
        self.g_sn = tk.Text(sn_frame, font=self.list_font, wrap="none", height=10)
        sn_y = ttk.Scrollbar(sn_frame, orient="vertical", command=self.g_sn.yview)
        sn_x = ttk.Scrollbar(sn_frame, orient="horizontal", command=self.g_sn.xview)
        self.g_sn.configure(yscrollcommand=sn_y.set, xscrollcommand=sn_x.set)
        self.g_sn.pack(fill='both', expand=True, padx=4, pady=4)
        sn_y.pack(side="right", fill="y")
        sn_x.pack(side="bottom", fill="x")

        tk.Label(de_frame, text="DE (selected in current list)", font=('Arial', 8, 'bold')).pack(anchor='w', padx=4, pady=(4, 0))
        self.g_de = tk.Text(de_frame, font=self.list_font, wrap="none", height=8)
        de_y = ttk.Scrollbar(de_frame, orient="vertical", command=self.g_de.yview)
        de_x = ttk.Scrollbar(de_frame, orient="horizontal", command=self.g_de.xview)
        self.g_de.configure(yscrollcommand=de_y.set, xscrollcommand=de_x.set)
        self.g_de.pack(fill='both', expand=True, padx=4, pady=4)
        de_y.pack(side="right", fill="y")
        de_x.pack(side="bottom", fill="x")

        self._tags(self.g_sn)
        self._tags(self.g_de)

        def init_sash():
            if self._gauge_sash_init:
                return
            h = paned.winfo_height()
            if h > 20:
                paned.sash_place(0, 0, int(h * 0.78))
                self._gauge_sash_init = True

        self.after_idle(init_sash)
        self.g_sn.bind('<Configure>', lambda e: self.update_stats(self.last_stats))
        self.g_de.bind('<Configure>', lambda e: self.update_stats(self.last_stats))

    def _apply_font_size(self):
        try:
            size = int(self.font_size_var.get())
        except (TypeError, ValueError):
            return
        size = max(8, min(18, size))
        self.list_font.configure(size=size)
        for widget in (self.txt_sn, self.txt_de, self.g_sn, self.g_de):
            widget.configure(font=self.list_font)

    def _tags(self, w):
        w.tag_config("normal", foreground="#1f2933")
        w.tag_config("active", foreground="#1b5e20")
        w.tag_config("dim", foreground="#9e9e9e")
        w.tag_config("unavail", foreground="#d0d0d0")
        w.tag_config("fill", background="#4caf50", foreground="#4caf50")
        w.tag_config("empty", background="#eeeeee", foreground="#eeeeee")
        w.tag_config("current_focus", background="#fff3e0", foreground="#bf360c")
        w.tag_config("current_selected", background="#e0f2f1", foreground="#00695c")
        w.tag_config("code_cell", background="#eef3ff")
        w.tag_config("value_cell", background="#fff7ec")

    def _click(self, e, code_type, w):
        idx = w.index(f"@{e.x},{e.y}")
        tags = w.tag_names(idx)
        code_tag = next((t for t in tags if t.startswith('code_tag:')), None)
        if not code_tag:
            return
        _, pane, code = code_tag.split(':', 2)
        if code == '-' or not code:
            return
        self.callbacks['toggle'](code_type, code)

    def _sorted_codes(self, codes):
        def sort_key(code):
            m = self._code_pattern.match(code)
            if m:
                prefix, num, suffix = m.groups()
                try:
                    num = int(num)
                except ValueError:
                    num = 0
                return (prefix, num, suffix)
            return (code, 0, "")
        return sorted(codes, key=sort_key)

    def update_stats(self, s):
        if not s: return
        self.last_stats = s
        current_selected = s.get('current_selected', False)

        sel_sn = s.get('sel_visible_sn', Counter())
        sel_de = s.get('sel_visible_de', Counter())
        vis_sn = s.get('visible_sn', Counter())
        vis_de = s.get('visible_de', Counter())

        self._draw_list(
            self.txt_sn,
            s['total_sn'],
            sel_sn,
            vis_sn,
            s['filters_sn'],
            s['available_sn'],
            s.get('current_sn_codes', set()),
            current_selected
        )
        self._draw_list(
            self.txt_de,
            s['total_de'],
            sel_de,
            vis_de,
            s['filters_de'],
            s['available_de'],
            s.get('current_de_codes', set()),
            current_selected
        )
        self._draw_gauge(self.g_sn, vis_sn, sel_sn)
        self._draw_gauge(self.g_de, vis_de, sel_de)


    def _draw_list(self, w, tot, sel, vis, act, avail, current_codes, highlight_selected):
        w.config(state=tk.NORMAL); w.delete('1.0', tk.END)
        entries = []
        max_code_len = 0
        max_value_len = 0
        pane_key = getattr(w, 'pane_key', '')
        use_percent = self.show_percent.get()
        for c in self._sorted_codes(tot.keys()):
            chk = "☑ " if c in act else "☐ "
            base_tag = "active" if c in act else ("unavail" if c not in avail else ("dim" if sel[c] == 0 else "normal"))
            tags = [base_tag]
            if c in current_codes:
                tags.append('current_selected' if highlight_selected else 'current_focus')
            tags = tuple(tags)
            sel_cnt = sel.get(c, 0)
            vis_cnt = vis.get(c, 0)
            if use_percent:
                pct = (sel_cnt / vis_cnt * 100) if vis_cnt else 0
                value = f"{pct:4.1f}%"
            else:
                value = f"{sel_cnt:>2}/{vis_cnt:<2}"
            code_text = f"{chk}{c:<5} "
            max_code_len = max(max_code_len, len(code_text))
            max_value_len = max(max_value_len, len(value))
            code_tag = f"code_tag:{pane_key}:{c}"
            entries.append((code_text, value, tags, code_tag, c))

        if not entries:
            w.config(state=tk.DISABLED)
            return

        col_width = max_code_len + max_value_len + 2
        for i in range(0, len(entries), 2):
            self._insert_code_block(w, entries[i], col_width, max_value_len)
            if i + 1 < len(entries):
                w.insert(tk.END, "  ")
                self._insert_code_block(w, entries[i + 1], col_width, max_value_len)
            w.insert(tk.END, "\n")
        w.config(state=tk.DISABLED)

    def _insert_code_block(self, widget, entry, col_width, max_value_len):
        code_text, value_text, tags, code_tag, code = entry
        widget.insert(tk.END, code_text, tags + (code_tag, 'code_cell'))
        value_part = value_text.rjust(max_value_len)
        widget.insert(tk.END, value_part, tags + ('value_cell',))
        pad = col_width - (len(code_text) + len(value_part))
        if pad > 0:
            widget.insert(tk.END, " " * pad, tags)

    def _draw_gauge(self, w, visible_counts, selected_counts):
        w.config(state=tk.NORMAL)
        w.delete('1.0', tk.END)

        if not visible_counts:
            w.config(state=tk.DISABLED)
            return

        width_px = w.winfo_width()
        if width_px <= 1:
            width_px = 520

        char_px = max(6, self.list_font.measure("0"))
        entries = []
        for c in self._sorted_codes(visible_counts.keys()):
            denom = int(visible_counts.get(c, 0))
            num = int(selected_counts.get(c, 0))
            ratio = (num / denom) if denom > 0 else 0.0
            entries.append((c, ratio, num, denom))

        for code, ratio, num, denom in entries:
            prefix = f"{code:<6} "
            suffix = f" {num}/{denom} {ratio*100:5.1f}%"
            reserve_px = self.list_font.measure(prefix + "[] " + suffix) + 16
            bar_len = max(8, min(80, int((width_px - reserve_px) / char_px)))
            filled = int(round(ratio * bar_len))
            filled = max(0, min(bar_len, filled))
            empty = bar_len - filled

            w.insert(tk.END, prefix, "normal")
            w.insert(tk.END, "[", "normal")
            w.insert(tk.END, " " * filled, "fill")
            w.insert(tk.END, " " * empty, "empty")
            w.insert(tk.END, "]", "normal")
            w.insert(tk.END, suffix + "\n", "normal")

        w.config(state=tk.DISABLED)

    def _insert_gauge_block(self, widget, entry, col_width, bar_len):
        code, filled, empty = entry
        prefix = f"{code:<6} "
        widget.insert(tk.END, prefix, "normal")
        widget.insert(tk.END, "[", "normal")
        widget.insert(tk.END, " " * filled, "fill")
        widget.insert(tk.END, " " * empty, "empty")
        widget.insert(tk.END, "]", "normal")
        pad = col_width - (len(prefix) + bar_len + 2)
        if pad > 0:
            widget.insert(tk.END, " " * pad, "normal")

class MainView(tk.Tk):
    def __init__(self, c):
        super().__init__(); self.c = c
        self.title("Narremgen GUI"); self.geometry(CFG.get('UI','window_size'))
        self.bind('<Control-s>', lambda e: c.save_batch()); self.bind('<Control-f>', lambda e: self.nav.entry_search.focus())

        # self.nb = ttk.Notebook(self); self.nb.pack(fill='both', expand=True)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Narremgen.TNotebook", background="#d1d5db", borderwidth=0)
        style.configure(
            "Narremgen.TNotebook.Tab",
            padding=(16, 8),
            background="#e5e7eb",
            foreground="#111827"
        )
        style.map(
            "Narremgen.TNotebook.Tab",
            background=[("selected", "#2563eb"), ("active", "#dbeafe")],
            foreground=[("selected", "#ffffff"), ("active", "#111827")],
            font=[("selected", ("Segoe UI", 10, "bold")), ("!selected", ("Segoe UI", 10))]
        )
        self.nb = ttk.Notebook(self, style="Narremgen.TNotebook")
        self.nb.pack(fill="both", expand=True)
        
        self.tab_wb = tk.Frame(self.nb)
        self.nb.add(self.tab_wb, text="🔬 SN/DE/K Studio")
        
        self.tab_gen = GeneratorTab(self.nb)
        self.nb.add(self.tab_gen, text="🏭 Advice Generator")
        
        self.tab_seg = tk.Frame(self.nb)
        self.nb.add(self.tab_seg, text="✂ Segmenter")
        self.segmenter_app = SimpleSegmenterApp(self.tab_seg)

        self._build_workbench(self.tab_wb, c)

    def _build_workbench(self, parent, controller):
        tb = tk.Frame(parent, padx=5, pady=5, bg='#cccccc'); tb.pack(side=tk.TOP, fill=tk.X)
        b = lambda t, cmd, bg=None: tk.Button(tb, text=t, command=cmd, bg=bg).pack(side=tk.LEFT, padx=5)
        b("➕ Generator", lambda: self.nb.select(self.tab_gen), '#e1bee7')
        b("✂ Segmenter", lambda: self.nb.select(self.tab_seg), '#bbdefb')
        b("📑 Copy x3", controller.copy_all, '#ffeb3b')
        b("💾 Export", controller.save_batch, '#4caf50')
        b("📥 Import", controller.import_sel, None)

        pm = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=8, bg="#999999"); pm.pack(fill='both', expand=True)
        pl = tk.PanedWindow(pm, orient=tk.VERTICAL, sashwidth=8, bg="#cccccc"); pm.add(pl, minsize=480)
        
        ncbs = {'nav_to': controller.nav_abs, 'toggle_sel': controller.toggle, 'search': controller.search}
        self.nav = NavigationPanel(pm, ncbs); pm.add(self.nav, minsize=220, width=260)
        
        self.dash = Dashboard(pl); pl.add(self.dash, minsize=40)
        top = tk.Frame(pl); pl.add(top, height=450); top.columnconfigure(0, weight=1); top.columnconfigure(1, weight=1); top.rowconfigure(0, weight=1)
        vcbs = {'load': controller.load, 'clear': controller.clear, 'toggle': controller.toggle, 'copy_one': controller.copy1, 'save_one': controller.save1}
        variant_font_default = _load_ui_font_size('ui_font_size_variants', 'ui_font_size_main', 'font_size_text')
        self.variant_font_size_var = tk.IntVar(value=variant_font_default)
        self.views = [
            VariantView(top, 0, vcbs, font_size_var=self.variant_font_size_var),
            VariantView(top, 1, vcbs, font_size_var=self.variant_font_size_var),
            VariantView(pl, 2, vcbs, font_size_var=self.variant_font_size_var),
        ]
        self.views[0].grid(row=0, column=0, sticky="nsew", padx=1); self.views[1].grid(row=0, column=1, sticky="nsew", padx=1)
        pl.add(self.views[2], stretch="always")
        self.variant_font_size_var.trace_add('write', self._on_variant_font_change)

        scbs = {'toggle': controller.toggle_filter, 'all_sn': controller.all_sn, 'none_sn': controller.none_sn,
                'all_de': controller.all_de, 'none_de': controller.none_de}
        self.stats = StatsPanel(pm, scbs); pm.add(self.stats, minsize=200, width=320)
        pm.paneconfigure(pl, stretch="always")
        pm.paneconfigure(self.nav, stretch="always")
        pm.paneconfigure(self.stats, stretch="always")

        ft = tk.Frame(parent, bg='#eeeeee', height=25); ft.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(ft, text="🗑️", command=controller.reset, font=('Arial',8), bd=0).pack(side=tk.RIGHT, padx=2)

    def refresh(self, m):
        st = m.get_detailed_stats()
        self.dash.update_stats(st)
        for i, v in enumerate(self.views): v.update_view(m.get_view_data(i))
        self.stats.update_stats(st)
        self.nav.update_list(m.get_nav_list_data(), m.current_absolute_index)

    def _on_variant_font_change(self, *_args):
        var = getattr(self, 'variant_font_size_var', None)
        if var is None:
            return
        try:
            size = int(var.get())
        except (tk.TclError, ValueError):
            return
        normalized = _clamp_font_size(size)
        if normalized != size:
            var.set(normalized)
            return
        try:
            CFG.set('UI', 'ui_font_size_variants', normalized)
            CFG.set('UI', 'ui_font_size_main', normalized)
        except Exception:
            _logger.warning("Unexpected error in MainView._on_variant_font_change", exc_info=True)

    def msg(self, t, m, err=False): (messagebox.showerror if err else messagebox.showinfo)(t, m)

# class Controller:
#     def __init__(self):
#         self.m = MergerModel()
#         self.v = None
#         self._autosave_path = CONFIG_ROOT / "autosave_selection.json"

#     def set_view(self, v): self.v = v; self.v.refresh(self.m); self.v.after(30000, self.autosave)

#     def autosave(self):
#         try:
#             self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(self._autosave_path, "w", encoding="utf-8") as f:
#                 json.dump(list(self.m.selected_indices), f)
#         except (OSError, TypeError):
#             _logger.warning("[MERGER] autosave failed", exc_info=True)
#         self.v.after(30000, self.autosave)


# def load(self, i): 
#     f = filedialog.askopenfilename()
#     if not f:
#         return
#     ok, msg = self.m.load_file_into_slot(i, f)
#     if ok:
#         self.v.refresh(self.m)
#     else:
#         self.v.msg("Erreur", msg, True)
#     def clear(self, i): self.m.clear_slot(i); self.v.refresh(self.m)
#     def nav_abs(self, i): 
#         if self.m.set_absolute_index(i): self.v.refresh(self.m)
#     def navigate(self, d): 
#         if self.m.move_index(d): self.v.refresh(self.m)
#     def toggle(self): self.m.toggle_selection_current(); self.v.refresh(self.m)
#     def search(self, q): self.m.set_search_query(q); self.v.refresh(self.m)
#     def toggle_filter(self, t, c):
#         (self.m.toggle_filter_sn if t=='SN' else self.m.toggle_filter_de)(c); self.v.refresh(self.m)
#     def all_sn(self): self.m.select_all_sn(); self.v.refresh(self.m)
#     def none_sn(self): self.m.select_none_sn(); self.v.refresh(self.m)
#     def all_de(self): self.m.select_all_de(); self.v.refresh(self.m)
#     def none_de(self): self.m.select_none_de(); self.v.refresh(self.m)
    
#     def copy1(self, i): 
#         c, _ = self.m.get_text_content(i)
#         if c: self.v.clipboard_clear(); self.v.clipboard_append(c); self.v.update()
#     def save1(self, i):
#         c, n = self.m.get_text_content(i)
#         if not c: return
#         f = filedialog.asksaveasfilename(initialfile=n, defaultextension=".txt")
#         if f: 
#             with open(f, "w", encoding="utf-8") as o: o.write(c)
#     def copy_all(self):
#         c = self.m.get_all_views_content()
#         if c: self.v.clipboard_clear(); self.v.clipboard_append(c); self.v.update()
    
#     def import_sel(self):
#         f = filedialog.askopenfilename()
#         if f: 
#             ok, m = self.m.import_selection(f)
#             if ok: self.v.refresh(self.m); self.v.msg("Info", m)
#             else: self.v.msg("Erreur", m, True)
    
#     def reset(self):
#         if messagebox.askyesno("RAZ", "Tout désélectionner ->"):
#             self.m.backup_selection(); self.m.reset_all_selections(); self.v.refresh(self.m)
            
#     def save_batch(self):
#         d = self.m.get_batch_export_data()
#         if not d: return self.v.msg("Erreur", "Rien de sélectionné", True)
#         tgt = filedialog.askdirectory()
#         if tgt:
#             fp = os.path.join(tgt, "SELECTED_HEADERS_LIST.txt")
#             if os.path.exists(fp): shutil.copy(fp, fp+".bak")
#             with open(fp, "w", encoding="utf-8") as f: f.write("\n".join(d['headers']))
#             for i, s in d['slots'].items():
#                 with open(os.path.join(tgt, f"{os.path.splitext(s['filename'])[0]}_SELECTED.txt"), "w", encoding="utf-8") as f: f.write(s['content'])
#             self.v.msg("Succès", "Sauvegardé.")

# if __name__ == '__main__':
#     c = Controller()
#     app = MainView(c)
#     c.set_view(app)
#     app.mainloop()

class Controller:
    def __init__(self):
        self.m = MergerModel()
        self.v = None
        self._autosave_path = CONFIG_ROOT / "autosave_selection.json"

    def set_view(self, v):
        self.v = v
        self.v.refresh(self.m)
        self.v.after(30000, self.autosave)

    def autosave(self):
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._autosave_path, "w", encoding="utf-8") as f:
                json.dump(list(self.m.selected_indices), f)
        except (OSError, TypeError):
            _logger.warning("[MERGER] autosave failed", exc_info=True)
        if self.v is not None:
            self.v.after(30000, self.autosave)

    def load(self, i):
        f = filedialog.askopenfilename()
        if not f:
            return
        ok, msg = self.m.load_file_into_slot(i, f)
        if ok:
            self.v.refresh(self.m)
        else:
            self.v.msg("Erreur", msg, True)

    def clear(self, i):
        self.m.clear_slot(i)
        self.v.refresh(self.m)

    def nav_abs(self, i):
        if self.m.set_absolute_index(i):
            self.v.refresh(self.m)

    def navigate(self, d):
        if self.m.move_index(d):
            self.v.refresh(self.m)

    def toggle(self):
        self.m.toggle_selection_current()
        self.v.refresh(self.m)

    def search(self, q):
        self.m.set_search_query(q)
        self.v.refresh(self.m)

    def toggle_filter(self, t, c):
        (self.m.toggle_filter_sn if t == 'SN' else self.m.toggle_filter_de)(c)
        self.v.refresh(self.m)

    def all_sn(self):
        self.m.select_all_sn()
        self.v.refresh(self.m)

    def none_sn(self):
        self.m.select_none_sn()
        self.v.refresh(self.m)

    def all_de(self):
        self.m.select_all_de()
        self.v.refresh(self.m)

    def none_de(self):
        self.m.select_none_de()
        self.v.refresh(self.m)

    def copy1(self, i):
        c, _ = self.m.get_text_content(i)
        if c:
            self.v.clipboard_clear()
            self.v.clipboard_append(c)
            self.v.update()

    def save1(self, i):
        c, n = self.m.get_text_content(i)
        if not c:
            return
        f = filedialog.asksaveasfilename(initialfile=n, defaultextension=".txt")
        if f:
            with open(f, "w", encoding="utf-8") as o:
                o.write(c)

    def copy_all(self):
        c = self.m.get_all_views_content()
        if c:
            self.v.clipboard_clear()
            self.v.clipboard_append(c)
            self.v.update()

    def import_sel(self):
        f = filedialog.askopenfilename()
        if not f:
            return
        ok, m = self.m.import_selection(f)
        if ok:
            self.v.refresh(self.m)
            self.v.msg("Info", m)
        else:
            self.v.msg("Erreur", m, True)

    def reset(self):
        if messagebox.askyesno("RAZ", "Tout désélectionner ->"):
            self.m.backup_selection()
            self.m.reset_all_selections()
            self.v.refresh(self.m)

    def save_batch(self):
        d = self.m.get_batch_export_data()
        if not d:
            self.v.msg("Erreur", "Rien de sélectionné", True)
            return
        tgt = filedialog.askdirectory()
        if not tgt:
            return
        fp = os.path.join(tgt, "SELECTED_HEADERS_LIST.txt")
        if os.path.exists(fp):
            shutil.copy2(fp, fp + ".bak")
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(d['headers']))
        for i, s in d['slots'].items():
            out_path = os.path.join(tgt, f"{os.path.splitext(s['filename'])[0]}_SELECTED.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(s['content'])
        self.v.msg("Succès", "Sauvegardé.")


if __name__ == '__main__':
    c = Controller()
    app = MainView(c)
    c.set_view(app)
    app.mainloop()
