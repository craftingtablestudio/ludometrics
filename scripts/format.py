"""Format all Python source files with ruff.

Usage:
    uv run format
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent.parent
    result = subprocess.run(["ruff", "format", str(root)], check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
