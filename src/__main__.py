"""
Narremgen CLI entrypoint for `python -m narremgen`.
"""

from __future__ import annotations

import os
import sys
import traceback

from narremgen.main import main as _main


def _debug_enabled() -> bool:
    v = os.environ.get("NARREMGEN_DEBUG", "")
    return v.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    try:
        rc = _main()
        return int(rc) if rc is not None else 0
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user.\n")
        return 130
    except Exception as exc:
        sys.stderr.write(
            "Narremgen project package: last operation did not complete.\n"
            "Some optional features may not be available in this build yet,"
            "or a missing optional dependency may be required to be installed.\n"
            "Please report the issue and the ran command on the project website.\n"
            f"Error: {exc.__class__.__name__}: {exc}\n"
        )
        if _debug_enabled():
            traceback.print_exc()
        else:
            sys.stderr.write("Tip: set NARREMGEN_DEBUG=1 to print the full traceback.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
