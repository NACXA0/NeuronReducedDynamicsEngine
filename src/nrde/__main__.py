"""Allow ``python -m nrde`` when the console script is not on PATH."""

from __future__ import annotations

import sys

from nrde.cli import main

if __name__ == "__main__":
    sys.exit(main())
